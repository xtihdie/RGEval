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
from evaluation.question_bank import load_converge_groups, load_question_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("LLM converge scoring (legacy DIM logic)")
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

    num_threads: int = 20
    tag: str = "2"

    agent_name: str = "zhipu"
    model_name: str = "gpt-4o-mini"

    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def question_score_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"
        )

    @property
    def result_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_converge_1.csv"
        )


RESULT_COLUMNS = ["课程编号", "维度", "级别", "AI评级", "评语"]

# ============================================================
# 原始业务常量（完全保留）
# ============================================================

RUBRIC_QUESTIONS = []
RUBRIC_QUESTIONS = load_question_dimensions(RUBRIC_QUESTIONS)

DIM = []
DIM = load_converge_groups(DIM)

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


def discover_codes(root: str) -> List[int]:
    return sorted(int(f.split("_")[0]) for f in os.listdir(root) if f.split("_")[0].isdigit())


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
        out[cid][dim] = f"：得分：[{r['级别']}]；评语：[{r.get('评语','')}]"
    return out


def classify_level(score: float) -> str:
    return "专家级" if score >= 66 else ("中级" if score >= 33 else "初级")


# ============================================================
# 核心：单任务（纯函数）
# ============================================================

def grade_single(
    lesson_id: int,
    dim: int,
    cfg: Config,
    score_df: pd.DataFrame,
    q_scores: Dict[int, List[str]],
    runner: ThreadRunner,
) -> Optional[Dict]:
    row = score_df[score_df["课例id"] == lesson_id]
    if row.empty or lesson_id not in q_scores:
        return None

    entries = q_scores[lesson_id]
    summary = "\n".join(
        f"标准({i}) {entries[k-1]}"
        for i, k in enumerate(DIM[dim], 1)
        if k - 1 < len(entries)
    )

    rubric = "\n".join(
        f"标准({i}) {RUBRIC_QUESTIONS[k-1]}"
        for i, k in enumerate(DIM[dim], 1)
    )

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
    except Exception:
        return None

    return {
        "课程编号": lesson_id,
        "维度": dim,
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

    all_codes = discover_codes(cfg.class_file_path)
    all_tasks = [(cid, d) for cid in all_codes for d in range(len(DIM))]

    if cfg.result_path.exists():
        result_df = read_csv_auto(cfg.result_path)
        done = {(int(r["课程编号"]), int(r["维度"])) for _, r in result_df.iterrows()}
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    pending = [t for t in all_tasks if t not in done]
    if not pending:
        return

    score_df = read_csv_auto(Path(cfg.score_file_path))
    q_scores = load_question_scores(cfg)

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    lock = threading.Lock()

    progress = build_score_progress()

    task_id = progress.add_task("progress", total=len(all_tasks), completed=len(done))

    with Live(progress, console=console):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    grade_single, cid, dim, cfg, score_df, q_scores, runner
                ): (cid, dim)
                for cid, dim in pending
            }

            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r:
                    with lock:
                        result_df.loc[len(result_df)] = r
                        result_df.to_csv(cfg.result_path, index=False, encoding=cfg.output_encoding)
                progress.advance(task_id)


if __name__ == "__main__":
    main()
