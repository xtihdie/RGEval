from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rich.console import Console
from rich.live import Live

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pool.runner import ThreadRunner
from test.prompts import scoring_rubric_mutual_cross
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_rubric_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI（沿用 4_0 风格）
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("step4_1_1 dimension mutual scoring")
    p.add_argument("--agent-a", type=str, default="deepseek")
    p.add_argument("--model-a", type=str, default="deepseek3.1")
    p.add_argument("--agent-b", type=str, default="zhipu")
    p.add_argument("--model-b", type=str, default="glm-4-flash")
    p.add_argument("--num-threads", type=int, default=10)
    add_path_override_argument(p, "--class-file-path", "class_file_path", "Override classroom/dialogue source directory.")
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_path_override_argument(p, "--prev-dim-dir", "prev_dim_dir", "Override previous step-1 dimension result directory.")
    add_question_bank_argument(p)
    return p.parse_args()

# ============================================================
# Rubric（维度级）
# ============================================================

RUBRIC_ITEMS: List[Dict[str, str]] = []
RUBRIC_ITEMS = load_rubric_dimensions(RUBRIC_ITEMS)

# ============================================================
# Config（沿用 4_0 的 normalize_pair 思路）
# ============================================================

@dataclass
class Config:
    class_file_path: Path = PROJECT_ROOT / "test/data/300class/origin"

    # step1 的多维度结果目录（你给的模板 tag=1）
    prev_dim_dir: Path = PROJECT_ROOT / "test/data/300class/results/1"

    # 4_1_1 输出目录
    result_dir: Path = PROJECT_ROOT / "test/data/300class/results/4/1"

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
        """保证 (agent_a, model_a) <= (agent_b, model_b)，A/B 交换复用同一输出文件"""
        a = (self.agent_a, self.model_a)
        b = (self.agent_b, self.model_b)
        if a > b:
            self.agent_a, self.model_a, self.agent_b, self.model_b = (
                self.agent_b, self.model_b, self.agent_a, self.model_a
            )

    @property
    def dim1_a_path(self) -> Path:
        return self.prev_dim_dir / f"1_{self._safe(self.agent_a)}_{self._safe(self.model_a)}_scores.csv"

    @property
    def dim1_b_path(self) -> Path:
        return self.prev_dim_dir / f"1_{self._safe(self.agent_b)}_{self._safe(self.model_b)}_scores.csv"

    @property
    def output_path(self) -> Path:
        return (
            self.result_dir
            / f"4_1_1_{self._safe(self.agent_a)}_{self._safe(self.model_a)}__"
              f"{self._safe(self.agent_b)}_{self._safe(self.model_b)}_scores_1.csv"
        )

# ============================================================
# CSV 工具（复用你 4_0 的写法）
# ============================================================

RESULT_COLUMNS = [
    "课程编号",
    "维度",
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
# 核心：单任务 (lesson_id, dim) 的维度互评 + 4_0 融合公式
# ============================================================

def mutual_score_one(
    lesson_id: int,
    dim: int,
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

    # 该维度的 rubric
    if dim < 0 or dim >= len(RUBRIC_ITEMS):
        raise RuntimeError(f"Invalid dim={dim} for lesson {lesson_id}")
    rubric = RUBRIC_ITEMS[dim]

    # step1 两个模型在该维度的原始分数+评语
    score_a = level_to_num(row["AI评级_a"])
    score_b = level_to_num(row["AI评级_b"])
    comment_a = str(row.get("评语_a", ""))
    comment_b = str(row.get("评语_b", ""))

    if pd.isna(score_a) or pd.isna(score_b):
        raise RuntimeError(f"Invalid base scores for lesson {lesson_id}, dim {dim}")

    # A 评 B：输入 B 的（该维度）分数+评语
    scores_result_b = f"评级：{row['AI评级_b']}\n评语：{comment_b}"
    msg_a = [[
        {"role": "system", "content": scoring_rubric_mutual_cross["system"]},
        {"role": "user", "content": scoring_rubric_mutual_cross["user"].format(
            class_conv=class_conv,
            criteria=rubric["criteria"],
            expound=rubric["expound"],
            scores_result=scores_result_b,
        )},
    ]]
    out_a = runner_a.run(msg_a)[0]
    a_scores = parse_scores(out_a)

    # B 评 A：输入 A 的（该维度）分数+评语
    scores_result_a = f"评级：{row['AI评级_a']}\n评语：{comment_a}"
    msg_b = [[
        {"role": "system", "content": scoring_rubric_mutual_cross["system"]},
        {"role": "user", "content": scoring_rubric_mutual_cross["user"].format(
            class_conv=class_conv,
            criteria=rubric["criteria"],
            expound=rubric["expound"],
            scores_result=scores_result_a,
        )},
    ]]
    out_b = runner_b.run(msg_b)[0]
    b_scores = parse_scores(out_b)

    if len(a_scores) != 3 or len(b_scores) != 3:
        raise RuntimeError(
            f"Expected 3 mutual scores each, got {len(a_scores)} and {len(b_scores)} "
            f"for lesson {lesson_id}, dim {dim}"
        )

    # 复用你 4_0 的融合公式：把“该维度的 base score”按互评占比融合
    weight = [0.35, 0.35, 0.3]
    final = sum(
        (score_a * (i / (i + j)) + score_b * (j / (i + j))) * w
        if (i + j) > 0 else ((score_a + score_b) / 2) * w
        for i, j, w in zip(a_scores, b_scores, weight)
    )

    return {
        "课程编号": lesson_id,
        "维度": dim,
        "AI评级": num_to_level(final),
    }

# ============================================================
# 主流程：断点续跑 + 边跑边落盘（完全沿用 4_0）
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
    override_config_value(cfg, args, "prev_dim_dir", as_path=True)
    override_config_value(cfg, args, "result_dir", as_path=True)
    cfg.normalize_pair()
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 step1 多维度评分结果（两个模型）
    if not cfg.dim1_a_path.exists():
        raise FileNotFoundError(f"Missing step1 dim score file: {cfg.dim1_a_path}")
    if not cfg.dim1_b_path.exists():
        raise FileNotFoundError(f"Missing step1 dim score file: {cfg.dim1_b_path}")

    df_a = read_csv_auto(cfg.dim1_a_path)
    df_b = read_csv_auto(cfg.dim1_b_path)

    # 只保留需要的列（防止多余列名冲突）
    need_cols = {"课程编号", "维度", "AI评级", "评语"}
    if not need_cols.issubset(set(df_a.columns)):
        raise RuntimeError(f"{cfg.dim1_a_path} missing columns: {need_cols - set(df_a.columns)}")
    if not need_cols.issubset(set(df_b.columns)):
        raise RuntimeError(f"{cfg.dim1_b_path} missing columns: {need_cols - set(df_b.columns)}")

    df_a = df_a[list(need_cols)].copy()
    df_b = df_b[list(need_cols)].copy()

    # 合并：key=(课程编号,维度)
    merged = df_a.merge(df_b, on=["课程编号", "维度"], suffixes=("_a", "_b"))

    # 任务集合：所有 (lesson_id, dim)
    all_tasks: List[Tuple[int, int]] = [
        (int(r["课程编号"]), int(r["维度"])) for _, r in merged.iterrows()
    ]

    # 断点续跑：如果已有输出，就读取 done 集合
    if cfg.output_path.exists():
        out_df = read_csv_auto(cfg.output_path)
        done: Set[Tuple[int, int]] = set(
            (int(r["课程编号"]), int(r["维度"])) for _, r in out_df.iterrows()
        )
        result_df = out_df
    else:
        done = set()
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)

    pending_tasks = [t for t in all_tasks if t not in done]

    if not pending_tasks:
        console.print("[bold green]✅ All tasks already done.[/bold green]")
        return

    # runner
    runner_a = ThreadRunner(cfg.agent_a, cfg.model_a, max_workers=cfg.num_threads)
    runner_b = ThreadRunner(cfg.agent_b, cfg.model_b, max_workers=cfg.num_threads)

    lock = threading.Lock()

    progress = build_score_progress()
    task_id = progress.add_task(
        "progress",
        total=len(all_tasks),
        completed=len(done),
    )

    merged_idx = merged.set_index(["课程编号", "维度"])

    with Live(progress, console=console, refresh_per_second=cfg.num_threads):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    mutual_score_one,
                    lesson_id,
                    dim,
                    merged_idx.loc[(lesson_id, dim)],
                    cfg,
                    runner_a,
                    runner_b,
                ): (lesson_id, dim)
                for (lesson_id, dim) in pending_tasks
            }

            for f in concurrent.futures.as_completed(futures):
                lesson_id, dim = futures[f]
                try:
                    r = f.result()
                except Exception as e:
                    console.print(f"[bold red]💥 Failed lesson {lesson_id}, dim {dim}: {e}[/bold red]")
                    progress.advance(task_id)
                    continue

                with lock:
                    result_df.loc[len(result_df)] = r
                    result_df.to_csv(cfg.output_path, index=False, encoding=cfg.output_encoding)

                progress.advance(task_id)

    console.print(f"[bold green]✅ Saved => {cfg.output_path}[/bold green]")

if __name__ == "__main__":
    main()
