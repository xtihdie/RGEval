from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from rich.console import Console
from rich.live import Live

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pool.runner import ThreadRunner
from test.prompts import scoring_mutual_cross
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_rubric_text
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("step4_0 raw mutual scoring")
    p.add_argument("--agent-a", type=str, default="deepseek")
    p.add_argument("--model-a", type=str, default="deepseek3.1")
    p.add_argument("--agent-b", type=str, default="zhipu")
    p.add_argument("--model-b", type=str, default="glm-4-flash")
    p.add_argument("--num-threads", type=int, default=10)
    add_path_override_argument(p, "--class-file-path", "class_file_path", "Override classroom/dialogue source directory.")
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_path_override_argument(p, "--prev-score-dir", "prev_score_dir", "Override previous step-0 result directory.")
    add_question_bank_argument(p)
    return p.parse_args()

# ============================================================
# RUBRIC（完整原样内嵌）
# ============================================================

RUBRIC: str = ""
RUBRIC = load_rubric_text(RUBRIC)

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    class_file_path: Path = PROJECT_ROOT / "test/data/300class/origin"
    # 保存到你要求的 results/4/0/*
    result_dir: Path = PROJECT_ROOT / "test/data/300class/results/4/0"
    # step0 的结果目录（0_score.py 输出）
    prev_score_dir: Path = PROJECT_ROOT / "test/data/300class/results/0"

    agent_a: str = "deepseek"
    model_a: str = "deepseek3.1"
    agent_b: str = "zhipu"
    model_b: str = "glm-4-flash"

    num_threads: int = 10
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    def normalize_pair(self) -> None:
        """保证 (agent_a, model_a) <= (agent_b, model_b)，这样 A/B 交换也能复用同一个输出文件"""
        a = (self.agent_a, self.model_a)
        b = (self.agent_b, self.model_b)
        if a > b:
            self.agent_a, self.model_a, self.agent_b, self.model_b = (
                self.agent_b, self.model_b, self.agent_a, self.model_a
            )

    @property
    def output_path(self) -> Path:
        return (
            self.result_dir
            / f"4_0_{self._safe(self.agent_a)}_{self._safe(self.model_a)}__"
              f"{self._safe(self.agent_b)}_{self._safe(self.model_b)}_scores.csv"
        )

    @property
    def score0_a_path(self) -> Path:
        return self.prev_score_dir / f"0_{self._safe(self.agent_a)}_{self._safe(self.model_a)}_scores.csv"

    @property
    def score0_b_path(self) -> Path:
        return self.prev_score_dir / f"0_{self._safe(self.agent_b)}_{self._safe(self.model_b)}_scores.csv"

# ============================================================
# CSV 读写（复用 step0 风格）
# ============================================================

RESULT_COLUMNS = [
    "课程编号",
    "AI评级",
]

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法识别文件编码: {path}")

def read_dialogue_by_code(lesson_id: int, root: Path) -> Optional[pd.DataFrame]:
    for f in os.listdir(root):
        if f.startswith(str(lesson_id)) and f.endswith(".csv"):
            return read_csv_auto(root / f)
    return None

def build_class_conv(df: pd.DataFrame) -> str:
    return "\n".join(
        f"{r.get('角色','')}：{r.get('内容','')}（{r.get('类别','')}）"
        for _, r in df.iterrows()
    )

def parse_scores(raw: str) -> List[int]:
    # 兼容 <score>85</score> 与 <score>85<score>
    nums = re.findall(r"<score>\s*\[?\s*(\d{1,3})", raw or "")
    return [int(x) for x in nums]

def safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")
    
LEVEL_TO_NUM = {
    "初级": 1.0,
    "中级": 2.0,
    "专家级": 3.0,
}

NUM_TO_LEVEL = {
    1: "初级",
    2: "中级",
    3: "专家级",
}

def level_to_num(v) -> float:
    return LEVEL_TO_NUM.get(str(v).strip(), float("nan"))

def num_to_level(v: float) -> str:
    if pd.isna(v):
        return ""
    # 看最接近哪个整数等级
    k = int(round(v))
    return NUM_TO_LEVEL.get(k, "")

# ============================================================
# 核心：单课互评（A->B 和 B->A）
# ============================================================

def mutual_score_one(
    lesson_id: int,
    row: pd.Series,
    cfg: Config,
    runner_a: ThreadRunner,
    runner_b: ThreadRunner,
) -> Dict:

    # 课堂原文
    df = read_dialogue_by_code(lesson_id, cfg.class_file_path)
    if df is None:
        raise RuntimeError(f"No classroom data for lesson {lesson_id}")
    class_conv = build_class_conv(df)

    # step0 两个模型的原始打分（分数+评语）
    score_a = level_to_num(row["AI评级_a"])
    score_b = level_to_num(row["AI评级_b"])
    comment_a = str(row.get("评语_a", ""))
    comment_b = str(row.get("评语_b", ""))

    if pd.isna(score_a) or pd.isna(score_b):
        raise RuntimeError(f"Invalid base scores for lesson {lesson_id}")

    # A 评 B：输入 B 的分数+评语
    scores_result_b = f"评级：{row['AI评级_b']}\n评语：{comment_b}"
    msg_a = [[
        {"role": "system", "content": scoring_mutual_cross["system"]},
        {"role": "user", "content": scoring_mutual_cross["user"].format(
            criteria=RUBRIC,
            class_conv=class_conv,
            scores_result=scores_result_b,
        )},
    ]]
    out_a = runner_a.run(msg_a)[0]
    a_scores = parse_scores(out_a)

    # B 评 A：输入 A 的分数+评语
    scores_result_a = f"评级：{row['AI评级_a']}\n评语：{comment_a}"
    msg_b = [[
        {"role": "system", "content": scoring_mutual_cross["system"]},
        {"role": "user", "content": scoring_mutual_cross["user"].format(
            criteria=RUBRIC,
            class_conv=class_conv,
            scores_result=scores_result_a,
        )},
    ]]
    out_b = runner_b.run(msg_b)[0]
    b_scores = parse_scores(out_b)

    if len(a_scores) != 3 or len(b_scores) != 3:
        raise RuntimeError(
            f"Expected 3 mutual scores each, got {len(a_scores)} and {len(b_scores)}"
        )

    # 你的原始公式：用互评分数作为“权重占比”融合两个 base score
    # i = A 对 B 的某项互评分；j = B 对 A 的对应互评分
    weight = [0.35, 0.35, 0.3]
    final = sum(
        (score_a * (i / (i + j)) + score_b * (j / (i + j))) * w
        if (i + j) > 0 else ((score_a + score_b) / 2) * w
        for i, j, w in zip(a_scores, b_scores, weight)
    )

    return {
        "课程编号": lesson_id,
        "AI评级": num_to_level(final),
    }

# ============================================================
# 主流程：断点续跑 + 边跑边落盘（step0 同款思路）
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)

    cfg = Config(
        agent_a=args.agent_a,
        model_a=args.model_a,
        agent_b=args.agent_b,
        model_b=args.model_b,
        num_threads=args.num_threads,
    )
    override_config_value(cfg, args, "class_file_path", as_path=True)
    override_config_value(cfg, args, "result_dir", as_path=True)
    override_config_value(cfg, args, "prev_score_dir", as_path=True)
    cfg.normalize_pair()
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 step0 结果（两个模型）
    if not cfg.score0_a_path.exists():
        raise FileNotFoundError(f"Missing step0 score file: {cfg.score0_a_path}")
    if not cfg.score0_b_path.exists():
        raise FileNotFoundError(f"Missing step0 score file: {cfg.score0_b_path}")

    score_a = read_csv_auto(cfg.score0_a_path)
    score_b = read_csv_auto(cfg.score0_b_path)

    # 合并后每行带 suffix：*_a, *_b
    merged = score_a.merge(score_b, on="课程编号", suffixes=("_a", "_b"))
    all_ids = set(merged["课程编号"].astype(int).tolist())

    # 断点续跑：如果已有输出，就读取 done 集合
    if cfg.output_path.exists():
        out_df = read_csv_auto(cfg.output_path)
        done: Set[int] = set(out_df["课程编号"].astype(int)).intersection(all_ids)
        result_df = out_df  # 继续 append
    else:
        done = set()
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)

    pending_ids = [i for i in sorted(all_ids) if i not in done]

    if not pending_ids:
        console.print("[bold green]✅ All lessons already done.[/bold green]")
        return

    # runner
    runner_a = ThreadRunner(cfg.agent_a, cfg.model_a, max_workers=cfg.num_threads)
    runner_b = ThreadRunner(cfg.agent_b, cfg.model_b, max_workers=cfg.num_threads)

    lock = threading.Lock()

    progress = build_score_progress()
    task_id = progress.add_task(
        "progress",
        total=len(all_ids),
        completed=len(done),
    )

    # 为了快速索引每个 lesson 的 row
    merged_idx = merged.set_index("课程编号")

    with Live(progress, console=console, refresh_per_second=cfg.num_threads):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    mutual_score_one,
                    lesson_id,
                    merged_idx.loc[lesson_id],
                    cfg,
                    runner_a,
                    runner_b,
                ): lesson_id
                for lesson_id in pending_ids
            }

            for f in concurrent.futures.as_completed(futures):
                lesson_id = futures[f]
                try:
                    r = f.result()
                except Exception as e:
                    # 不让一个失败拖垮全局（符合你“断点续跑”的核心哲学）
                    console.print(f"[bold red]💥 Failed lesson {lesson_id}: {e}[/bold red]")
                    progress.advance(task_id)
                    continue

                with lock:
                    result_df.loc[len(result_df)] = r
                    # 每成功一条就落盘：保证中途崩也能继续
                    result_df.to_csv(cfg.output_path, index=False, encoding=cfg.output_encoding)

                progress.advance(task_id)

    console.print(f"[bold green]✅ Saved => {cfg.output_path}[/bold green]")

if __name__ == "__main__":
    main()
