"""
Course-level convergence scoring pipeline.

延续 score/0_score.py 的统一模板结构，对单个课程的 6 个聚合维度结果
进行二次 LLM 整体评价。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.live import Live

# ============================================================
# 项目路径（统一模板）
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pool.runner import ThreadRunner
from test.prompts import converge_scoring_rubric
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_rubric_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Course-level converge scoring")
    p.add_argument("--agent", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--num-threads", dest="num_threads", type=int, default=None, help="Override max worker count for this stage.")
    add_path_override_argument(p, "--class-file-path", "class_file_path", "Override classroom/dialogue source directory.")
    add_path_override_argument(p, "--score-file-path", "score_file_path", "Override gold score CSV path.")
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_question_bank_argument(p)
    return p.parse_args()


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    class_file_path: str = str(PROJECT_ROOT / "test/data/300class/origin")
    score_file_path: str = str(PROJECT_ROOT / "test/data/300class/score.csv")
    result_dir: str = str(PROJECT_ROOT / "test/data/300class/results")

    num_threads: int = 4
    tag: str = "2"

    agent_name: str = "qwen"
    model_name: str = "Qwen/Qwen3-8B"

    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def question_score_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_converge_1.csv"
        )

    @property
    def result_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_converge_2.csv"
        )


RESULT_COLUMNS = ["课程编号", "级别", "AI评级", "评语"]

# ============================================================
# Rubric（保持业务原文）
# ============================================================

RUBRIC_SECTIONS: List[Dict[str, str]] = []
RUBRIC_SECTIONS = load_rubric_dimensions(RUBRIC_SECTIONS)


def build_rubric_prompt() -> str:
    return "\n\n".join(
        f"标准({i}) {r['criteria']}:\n{r['expound']}"
        for i, r in enumerate(RUBRIC_SECTIONS, 1)
    )


# ============================================================
# 工具函数
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"Cannot read {path}")


def discover_all_codes(root: str) -> List[int]:
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(root)
        if f.split("_")[0].isdigit()
    )


LEVELS = ("专家级", "中级", "初级")

def parse_llm_response(raw: str):
    # ---------- 1. 标准 <score>...</score> ----------
    m = re.search(
        r"<score[^>]*>\s*\[?\s*(专家级|中级|初级)\s*\]?\s*</score>",
        raw,
        re.I | re.S
    )
    if m:
        level = m.group(1)
    else:
        # ---------- 2. 非标准 <score/>中级<score/> ----------
        m = re.search(
            r"<score[^/>]*/?>\s*\[?\s*(专家级|中级|初级)\s*\]?\s*(?:</?score[^>]*>)?",
            raw,
            re.I | re.S
        )
        if m:
            level = m.group(1)
        else:
            # ---------- 3. 兜底：全文关键词 ----------
            level = None
            for lv in LEVELS:
                if lv in raw:
                    level = lv
                    break

    if not level:
        raise ValueError(f"Cannot parse level from response:\n{raw}")

    # ---------- comment（同样宽松） ----------
    cm = re.search(
        r"<comment[^>]*>\s*\[?(.*?)\]?\s*</comment>",
        raw,
        re.S
    )
    comment = cm.group(1).strip() if cm else ""

    return level, comment


def classify_level(score: float) -> str:
    total = 3 * score / 100
    if total >= 2:
        return "专家级"
    if total >= 1:
        return "中级"
    return "初级"


def load_question_scores(cfg: Config) -> Dict[int, List[str]]:
    if not cfg.question_score_path.exists():
        return {}

    df = read_csv_auto(cfg.question_score_path).sort_values(["课程编号", "维度"])
    out: Dict[int, List[str]] = {}

    for _, r in df.iterrows():
        cid, dim = int(r["课程编号"]), int(r["维度"])
        out.setdefault(cid, [])
        while len(out[cid]) <= dim:
            out[cid].append("")
        out[cid][dim] = f"标准({dim+1})：得分：[{r['级别']}]；评语：[{r.get('评语','')}]"

    return out


# ============================================================
# 核心：单课程整体评分（纯函数）
# ============================================================

def grade_single_lesson(
    lesson_id: int,
    cfg: Config,
    score_df: pd.DataFrame,
    aggregated_scores: Dict[int, List[str]],
    runner: ThreadRunner,
) -> Optional[Dict]:

    row = score_df[score_df["课例id"] == lesson_id]
    if row.empty or lesson_id not in aggregated_scores:
        return None

    summary = "\n".join(aggregated_scores[lesson_id])
    rubric = build_rubric_prompt()

    messages = [[
        {
            "role": "system",
            "content": converge_scoring_rubric["system"].format(criteria=rubric),
        },
        {
            "role": "user",
            "content": converge_scoring_rubric["user"].format(scores_result=summary),
        },
    ]]

    try:
        out = runner.run(messages)
        ai_level, comment = parse_llm_response(out[0])
    except Exception as e:
        print(e)
        return None

    return {
        "课程编号": lesson_id,
        "级别": row.iloc[0].get("级别", ""),
        "AI评级": ai_level,
        "评语": comment,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)
    cfg = Config(
        agent_name=args.agent or Config.agent_name,
        model_name=args.model or Config.model_name,
    )
    override_config_value(cfg, args, "num_threads")
    override_config_value(cfg, args, "class_file_path")
    override_config_value(cfg, args, "score_file_path")
    override_config_value(cfg, args, "result_dir")

    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)

    score_df = read_csv_auto(Path(cfg.score_file_path))
    aggregated_scores = load_question_scores(cfg)
    all_codes = discover_all_codes(cfg.class_file_path)

    if cfg.result_path.exists():
        result_df = read_csv_auto(cfg.result_path)
        done = set(result_df["课程编号"].astype(int))
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    pending = [c for c in all_codes if c not in done]
    if not pending:
        return

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    lock = threading.Lock()

    progress = build_score_progress()

    task_id = progress.add_task(
        "progress", total=len(all_codes), completed=len(done)
    )

    with Live(progress, console=console):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    grade_single_lesson,
                    cid,
                    cfg,
                    score_df,
                    aggregated_scores,
                    runner,
                ): cid
                for cid in pending
            }

            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r:
                    with lock:
                        result_df.loc[len(result_df)] = r
                        result_df.to_csv(
                            cfg.result_path,
                            index=False,
                            encoding=cfg.output_encoding,
                        )
                progress.advance(task_id)


if __name__ == "__main__":
    main()
