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

from llm_pool.runner import ThreadRunner  # noqa
from test.prompts import scoring_mutual_cross  # noqa
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_rubric_text
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("step4_2_3 mutual scoring (no-merge)")
    p.add_argument("--agent-a", type=str, default="deepseek")
    p.add_argument("--model-a", type=str, default="deepseek3.1")
    p.add_argument("--agent-b", type=str, default="zhipu")
    p.add_argument("--model-b", type=str, default="glm-4-flash")
    p.add_argument("--num-threads", type=int, default=10)
    add_path_override_argument(p, "--class-file-path", "class_file_path", "Override classroom/dialogue source directory.")
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_question_bank_argument(p)
    return p.parse_args()

# ============================================================
# RUBRIC（与 4_0 一致）
# ============================================================

RUBRIC: str = ""
RUBRIC = load_rubric_text(RUBRIC, key="rubric_text_compact")

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    class_file_path: Path = PROJECT_ROOT / "test/data/300class/origin"
    result_dir: Path = PROJECT_ROOT / "test/data/300class/results/4/2"

    agent_a: str = "deepseek"
    model_a: str = "deepseek3.1"
    agent_b: str = "zhipu"
    model_b: str = "glm-4-flash"

    num_threads: int = 10
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    def normalize_pair(self):
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
            / f"4_2_3_{self._safe(self.agent_a)}_{self._safe(self.model_a)}__"
              f"{self._safe(self.agent_b)}_{self._safe(self.model_b)}_scores_3.csv"
        )

    @property
    def converge_a_path(self) -> Path:
        return (
            self.result_dir
            / f"4_2_{self._safe(self.agent_a)}_{self._safe(self.model_a)}_"
              f"{self._safe(self.agent_a)}_{self._safe(self.model_a)}__"
              f"{self._safe(self.agent_b)}_{self._safe(self.model_b)}_converge_2.csv"
        )

    @property
    def converge_b_path(self) -> Path:
        return (
            self.result_dir
            / f"4_2_{self._safe(self.agent_b)}_{self._safe(self.model_b)}_"
              f"{self._safe(self.agent_a)}_{self._safe(self.model_a)}__"
              f"{self._safe(self.agent_b)}_{self._safe(self.model_b)}_converge_2.csv"
        )

# ============================================================
# CSV 工具
# ============================================================

RESULT_COLUMNS = [
    "课程编号",
    "final_score",
    "a_score_standard_fit",
    "a_score_consistency_fairness",
    "a_score_accuracy_constructive",
    "b_score_standard_fit",
    "b_score_consistency_fairness",
    "b_score_accuracy_constructive",
]

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                engine="python",
                on_bad_lines="warn",
            )
        except Exception:
            continue
    raise ValueError(f"Cannot read csv: {path}")

def load_converge2_map(path: Path) -> Dict[int, Dict]:
    df = read_csv_auto(path)
    need = {"课程编号", "分数", "评语"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)}")

    if not df["课程编号"].is_unique:
        raise RuntimeError(f"{path} has duplicated lesson_id")

    return {
        int(r["课程编号"]): {
            "score": float(r["分数"]),
            "comment": str(r.get("评语", "")),
        }
        for _, r in df.iterrows()
    }

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
    nums = re.findall(r"<score>\s*\[?\s*(\d{1,3})", raw or "")
    return [int(x) for x in nums]

# ============================================================
# 核心：单课互评
# ============================================================

def mutual_score_one(
    lesson_id: int,
    a_item: Dict,
    b_item: Dict,
    cfg: Config,
    runner_a: ThreadRunner,
    runner_b: ThreadRunner,
) -> Dict:

    df = read_dialogue_by_code(lesson_id, cfg.class_file_path)
    if df is None:
        raise RuntimeError(f"No classroom data for lesson {lesson_id}")

    class_conv = build_class_conv(df)

    score_a = a_item["score"]
    score_b = b_item["score"]
    comment_a = a_item["comment"]
    comment_b = b_item["comment"]

    # A 评 B
    msg_a = [[
        {"role": "system", "content": scoring_mutual_cross["system"]},
        {"role": "user", "content": scoring_mutual_cross["user"].format(
            criteria=RUBRIC,
            class_conv=class_conv,
            scores_result=f"分数：{score_b}\n评语：{comment_b}",
        )},
    ]]
    out_a = runner_a.run(msg_a)[0]
    a_scores = parse_scores(out_a)

    # B 评 A
    msg_b = [[
        {"role": "system", "content": scoring_mutual_cross["system"]},
        {"role": "user", "content": scoring_mutual_cross["user"].format(
            criteria=RUBRIC,
            class_conv=class_conv,
            scores_result=f"分数：{score_a}\n评语：{comment_a}",
        )},
    ]]
    out_b = runner_b.run(msg_b)[0]
    b_scores = parse_scores(out_b)

    if len(a_scores) != 3 or len(b_scores) != 3:
        raise RuntimeError("Mutual scores must be length 3")

    weight = [0.35, 0.35, 0.3]
    final = sum(
        (score_a * (i / (i + j)) + score_b * (j / (i + j))) * w
        if (i + j) > 0 else ((score_a + score_b) / 2) * w
        for i, j, w in zip(a_scores, b_scores, weight)
    )

    return {
        "课程编号": lesson_id,
        "final_score": final,
        "a_score_standard_fit": a_scores[0],
        "a_score_consistency_fairness": a_scores[1],
        "a_score_accuracy_constructive": a_scores[2],
        "b_score_standard_fit": b_scores[0],
        "b_score_consistency_fairness": b_scores[1],
        "b_score_accuracy_constructive": b_scores[2],
    }

# ============================================================
# 主流程：断点续跑 + 进度条
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
    cfg.normalize_pair()
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    conv_a_map = load_converge2_map(cfg.converge_a_path)
    conv_b_map = load_converge2_map(cfg.converge_b_path)

    all_ids = sorted(set(conv_a_map) & set(conv_b_map))

    if cfg.output_path.exists():
        out_df = read_csv_auto(cfg.output_path)
        done: Set[int] = set(out_df["课程编号"].astype(int))
        result_df = out_df
    else:
        done = set()
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)

    pending = [i for i in all_ids if i not in done]
    if not pending:
        console.print("[bold green]✅ All lessons already done.[/bold green]")
        return

    runner_a = ThreadRunner(cfg.agent_a, cfg.model_a, max_workers=cfg.num_threads)
    runner_b = ThreadRunner(cfg.agent_b, cfg.model_b, max_workers=cfg.num_threads)

    lock = threading.Lock()

    progress = build_score_progress()
    task_id = progress.add_task("progress", total=len(all_ids), completed=len(done))

    with Live(progress, console=console, refresh_per_second=8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    mutual_score_one,
                    lesson_id,
                    conv_a_map[lesson_id],
                    conv_b_map[lesson_id],
                    cfg,
                    runner_a,
                    runner_b,
                ): lesson_id
                for lesson_id in pending
            }

            for f in concurrent.futures.as_completed(futures):
                lesson_id = futures[f]
                try:
                    r = f.result()
                except Exception as e:
                    console.print(f"[bold red]💥 Failed lesson {lesson_id}: {e}[/bold red]")
                    progress.advance(task_id)
                    continue

                with lock:
                    result_df.loc[len(result_df)] = r
                    result_df.to_csv(cfg.output_path, index=False, encoding=cfg.output_encoding)

                progress.advance(task_id)

    console.print(f"[bold green]✅ Saved => {cfg.output_path}[/bold green]")

if __name__ == "__main__":
    main()
