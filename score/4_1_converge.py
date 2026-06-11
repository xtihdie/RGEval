from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from rich.console import Console
from rich.live import Live

# ============================================================
# 项目路径
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
    p = argparse.ArgumentParser("step4_1 converge scoring (lesson-level)")
    p.add_argument("--agent", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num-threads", type=int, default=10)
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_path_override_argument(p, "--step1-dir", "step1_dir", "Override step-1 result directory.")
    add_question_bank_argument(p)
    return p.parse_args()

# ============================================================
# Rubric
# ============================================================

RUBRIC_ITEMS: List[Dict[str, str]] = []
RUBRIC_ITEMS = load_rubric_dimensions(RUBRIC_ITEMS)

def build_rubric_text() -> str:
    return "\n\n".join(
        f"标准({i + 1}) {item['criteria']}:\n{item['expound']}"
        for i, item in enumerate(RUBRIC_ITEMS)
    )

RUBRIC_TEXT = build_rubric_text()

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    result_dir: Path = PROJECT_ROOT / "test/data/300class/results/4/1"
    step1_dir: Path = PROJECT_ROOT / "test/data/300class/results/1"
    agent_name: str = ""
    model_name: str = ""
    num_threads: int = 10
    output_encoding: str = "utf-8-sig"

# ============================================================
# Utils
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read csv: {path}")

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

STEP1_PAIR_RE = re.compile(
    r"^4_1_1_(?P<a_agent>.+?)_(?P<a_model>.+?)__(?P<b_agent>.+?)_(?P<b_model>.+?)_scores_1\.csv$"
)

def parse_pair_from_step1_filename(name: str):
    m = STEP1_PAIR_RE.match(name)
    if not m:
        raise ValueError(name)
    return (
        (m.group("a_agent"), m.group("a_model")),
        (m.group("b_agent"), m.group("b_model")),
    )

def converge_output_name(converger, pair):
    (ca, cm) = converger
    (a, b) = pair
    return f"4_1_{ca}_{cm}_{a[0]}_{a[1]}__{b[0]}_{b[1]}_converge.csv"

# ============================================================
# Data loaders
# ============================================================

def load_step1_dimension_summary(path: Path) -> Dict[int, str]:
    df = read_csv_auto(path)
    out = {}
    for cid, g in df.groupby("课程编号"):
        lines = []
        for _, r in g.sort_values("维度").iterrows():
            idx = int(r["维度"]) + 1
            lines.append(f"标准({idx})：得分[{r['AI评级']}]；评语[{r['评语']}]")
        out[int(cid)] = "\n".join(lines)
    return out

def load_mutual_evidence_by_lesson(df: pd.DataFrame) -> Dict[int, str]:
    out = {}
    for cid, g in df.groupby("课程编号"):
        lines = []
        for _, r in g.sort_values("维度").iterrows():
            idx = int(r["维度"]) + 1
            lines.append(
                f"标准({idx})：互评综合等级[{r['AI评级']}]"
            )
        out[int(cid)] = "\n".join(lines)
    return out

# ============================================================
# Converge worker
# ============================================================

def converge_one_lesson(
    lesson_id: int,
    converger,
    pair_name,
    origin_dim_summary,
    mutual_evidence,
    runner: ThreadRunner,
):
    try:
        text = (
            f"pair_file: {pair_name}\n"
            f"converger: {converger[0]}_{converger[1]}\n\n"
            f"{origin_dim_summary}\n\n{mutual_evidence}"
        )
        messages = [[
            {"role": "system", "content": converge_scoring_rubric["system"].format(criteria=RUBRIC_TEXT)},
            {"role": "user", "content": converge_scoring_rubric["user"].format(scores_result=text)},
        ]]
        out = runner.run(messages)[0]
        ai_level, comment = parse_llm_response(out)
        return {
            "课程编号": lesson_id,
            "AI评级": ai_level,
            "评语": comment,
        }
    except Exception:
        return None

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)
    cfg = Config(
        agent_name=args.agent,
        model_name=args.model,
        num_threads=args.num_threads,
    )
    override_config_value(cfg, args, "result_dir", as_path=True)
    override_config_value(cfg, args, "step1_dir", as_path=True)

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)

    all_pair_files = sorted(cfg.result_dir.glob("4_1_1_*_scores_1.csv"))
    step1_pair_files = []

    for p in all_pair_files:
        try:
            pair = parse_pair_from_step1_filename(p.name)
        except ValueError:
            continue
        if (cfg.agent_name, cfg.model_name) in pair:
            step1_pair_files.append(p)

    if not step1_pair_files:
        console.print("[red]No matching pair files[/red]")
        return

    for path in step1_pair_files:
        pair = parse_pair_from_step1_filename(path.name)
        converger = (cfg.agent_name, cfg.model_name)
        if converger not in pair:
            continue

        df_pair = read_csv_auto(path)
        mutual = load_mutual_evidence_by_lesson(df_pair)
        origin = load_step1_dimension_summary(
            cfg.step1_dir / f"1_{converger[0]}_{converger[1]}_scores.csv"
        )

        out_path = cfg.result_dir / converge_output_name(converger, pair)

        if out_path.exists():
            out_df = read_csv_auto(out_path)
            done: Set[int] = set(out_df["课程编号"])
            result_df = out_df
        else:
            done = set()
            result_df = pd.DataFrame(columns=["课程编号", "AI评级", "评语"])

        pending = [cid for cid in mutual if cid not in done and cid in origin]
        if not pending:
            continue

        lock = threading.Lock()
        progress = build_score_progress()
        task_id = progress.add_task("run", total=len(pending))

        with Live(progress, console=console):
            with concurrent.futures.ThreadPoolExecutor(cfg.num_threads) as exe:
                futures = {
                    exe.submit(
                        converge_one_lesson,
                        cid,
                        converger,
                        path.name,
                        origin[cid],
                        mutual[cid],
                        runner,
                    ): cid
                    for cid in pending
                }
                for f in concurrent.futures.as_completed(futures):
                    r = f.result()
                    if r:
                        with lock:
                            result_df.loc[len(result_df)] = r
                            result_df.to_csv(out_path, index=False, encoding=cfg.output_encoding)
                    progress.advance(task_id)

        console.print(f"[green]Saved {out_path}[/green]")

if __name__ == "__main__":
    main()
