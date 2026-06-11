from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from evaluation.question_bank import load_converge_groups, load_question_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("step4_2_1 converge scoring (DIM: 21 -> 6)")
    p.add_argument("--agent", type=str, default="deepseek")
    p.add_argument("--model", type=str, default="deepseek3.1")
    p.add_argument("--num-threads", type=int, default=10)
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_path_override_argument(p, "--step2-dir", "step2_dir", "Override step-2 result directory.")
    add_question_bank_argument(p)
    return p.parse_args()

# ============================================================
# 常量
# ============================================================

RUBRIC_QUESTIONS = []
RUBRIC_QUESTIONS = load_question_dimensions(RUBRIC_QUESTIONS)

DIM = []
DIM = load_converge_groups(DIM)

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    result_dir: Path = PROJECT_ROOT / "test/data/300class/results/4/2"
    step2_dir: Path = PROJECT_ROOT / "test/data/300class/results/2"
    agent_name: str = "deepseek"
    model_name: str = "deepseek3.1"
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

def parse_llm_response(raw: str) -> Tuple[float, str]:
    s = re.search(r"<score>\s*\[?(\d+(?:\.\d+)?)\]?\s*</score>", raw)
    c = re.search(r"<comment>(.*?)</comment>", raw, re.S)
    if not s:
        raise ValueError("No <score>")
    return float(s.group(1)), c.group(1).strip() if c else ""

STEP4_2_1_PAIR_RE = re.compile(
    r"^4_2_1_(?P<a_agent>.+?)_(?P<a_model>.+?)__(?P<b_agent>.+?)_(?P<b_model>.+?)_scores_1\.csv$"
)

def parse_pair_from_filename(name: str):
    m = STEP4_2_1_PAIR_RE.match(name)
    if not m:
        raise ValueError(name)
    return (
        (m.group("a_agent"), m.group("a_model")),
        (m.group("b_agent"), m.group("b_model")),
    )

def converge_output_name(converger, pair):
    (ca, cm) = converger
    (a, b) = pair
    return f"4_2_1_{ca}_{cm}_{a[0]}_{a[1]}__{b[0]}_{b[1]}_converge_1.csv"

def build_rubric(dim6: int) -> str:
    return "\n".join(
        f"标准({i}) {RUBRIC_QUESTIONS[k - 1]}"
        for i, k in enumerate(DIM[dim6], 1)
    )

# ============================================================
# Data loaders
# ============================================================

def load_step2_scores(agent: str, model: str) -> Dict[int, List[str]]:
    path = PROJECT_ROOT / "test/data/300class/results/2" / f"2_{agent}_{model}_scores.csv"
    df = read_csv_auto(path).sort_values(["课程编号", "维度"])
    out: Dict[int, List[str]] = {}
    for _, r in df.iterrows():
        cid = int(r["课程编号"])
        d = int(r["维度"])
        out.setdefault(cid, [])
        while len(out[cid]) <= d:
            out[cid].append("")
        out[cid][d] = f"：得分：[{r['分数']}]；评语：[{r.get('评语','')}]"
    return out

def load_mutual_21_to_6(path: Path) -> Dict[Tuple[int, int], str]:
    df = read_csv_auto(path)
    per_q = {}
    for _, r in df.iterrows():
        cid = int(r["课程编号"])
        q = int(r["维度"]) + 1
        per_q[(cid, q)] = (
            f"A→B=[{r['a_score_standard_fit']},{r['a_score_consistency_fairness']},{r['a_score_accuracy_constructive']}]；"
            f"B→A=[{r['b_score_standard_fit']},{r['b_score_consistency_fairness']},{r['b_score_accuracy_constructive']}]"
        )

    out = {}
    for cid in sorted(df["课程编号"].astype(int).unique()):
        for dim6 in range(6):
            lines = []
            for i, q in enumerate(DIM[dim6], 1):
                ev = per_q.get((cid, q), "[missing]")
                lines.append(f"标准({i}) {ev}")
            out[(cid, dim6)] = "\n".join(lines)
    return out

# ============================================================
# Worker
# ============================================================

def grade_one(
    cid: int,
    dim6: int,
    converger,
    q_scores,
    new_ev,
    runner,
):
    try:
        summary = "\n".join(
            f"标准({i}) {q_scores[cid][k - 1]}"
            for i, k in enumerate(DIM[dim6], 1)
            if (k - 1) < len(q_scores[cid])
        )

        text = (
            f"converger: {converger[0]}_{converger[1]}\n"
            f"lesson: {cid}\n"
            f"dim6: {dim6}\n\n"
            f"{summary}\n\n{new_ev[(cid, dim6)]}"
        )

        messages = [[
            {"role": "system", "content": converge_scoring_rubric["system"].format(criteria=build_rubric(dim6))},
            {"role": "user", "content": converge_scoring_rubric["user"].format(scores_result=text)},
        ]]

        out = runner.run(messages)[0]
        score, comment = parse_llm_response(out)

        return {"课程编号": cid, "维度": dim6, "分数": score, "评语": comment}
    except Exception:
        return None

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)
    cfg = Config(agent_name=args.agent, model_name=args.model, num_threads=args.num_threads)
    override_config_value(cfg, args, "result_dir", as_path=True)
    override_config_value(cfg, args, "step2_dir", as_path=True)

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)

    all_files = sorted(cfg.result_dir.glob("4_2_1_*_scores_1.csv"))
    pair_files = []

    for f in all_files:
        try:
            pair = parse_pair_from_filename(f.name)
        except ValueError:
            continue
        if (cfg.agent_name, cfg.model_name) in pair:
            pair_files.append(f)

    if not pair_files:
        console.print("[red]No matching mutual files[/red]")
        return

    for path in pair_files:
        pair = parse_pair_from_filename(path.name)
        converger = (cfg.agent_name, cfg.model_name)

        q_scores = load_step2_scores(*converger)
        new_ev = load_mutual_21_to_6(path)

        out_path = cfg.result_dir / converge_output_name(converger, pair)

        if out_path.exists():
            df = read_csv_auto(out_path)
            done = {(int(r["课程编号"]), int(r["维度"])) for _, r in df.iterrows()}
            result_df = df
        else:
            done = set()
            result_df = pd.DataFrame(columns=["课程编号", "维度", "分数", "评语"])

        pending = [k for k in new_ev if k not in done and k[0] in q_scores]
        if not pending:
            continue

        lock = threading.Lock()
        progress = build_score_progress()
        task_id = progress.add_task("run", total=len(pending))

        with Live(progress, console=console):
            with concurrent.futures.ThreadPoolExecutor(cfg.num_threads) as exe:
                futures = {
                    exe.submit(
                        grade_one,
                        cid,
                        dim6,
                        converger,
                        q_scores,
                        new_ev,
                        runner,
                    ): (cid, dim6)
                    for (cid, dim6) in pending
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
