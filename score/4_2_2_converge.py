from __future__ import annotations

import argparse
import concurrent.futures
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

from llm_pool.runner import ThreadRunner  # noqa: E402
from test.prompts import converge_scoring_rubric  # noqa: E402
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_rubric_dimensions  # noqa: E402
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI（默认值必须可直接运行）
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("step4_2_2 converge scoring")
    p.add_argument("--agent", type=str, default="deepseek")
    p.add_argument("--model", type=str, default="deepseek3.1")
    p.add_argument("--num-threads", type=int, default=10)
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_path_override_argument(p, "--step2-dir", "step2_dir", "Override step-2 result directory.")
    add_question_bank_argument(p)
    return p.parse_args()

# ============================================================
# Rubric（6大维度，用于 converge_scoring_rubric.system 的 criteria）
# ============================================================

RUBRIC_ITEMS: List[Dict[str, str]] = []
RUBRIC_ITEMS = load_rubric_dimensions(RUBRIC_ITEMS)

def build_rubric_text() -> str:
    return "\n\n".join(
        f"标准({i + 1}) {item['criteria']}:\n\t{item['expound']}"
        for i, item in enumerate(RUBRIC_ITEMS)
    )

RUBRIC_TEXT = build_rubric_text()

# ============================================================
# Config（不要改目录结构）
# ============================================================

@dataclass
class Config:
    # 4_2 的所有输出都在这里
    result_dir: Path = PROJECT_ROOT / "test/data/300class/results/4/2"
    # step2（2_score）结果目录：用于取 converge_2 的评语（原样照搬）
    step2_dir: Path = PROJECT_ROOT / "test/data/300class/results/2"

    agent_name: str = "deepseek"
    model_name: str = "deepseek3.1"
    num_threads: int = 30
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

# ============================================================
# 读写工具
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read csv: {path}")

def parse_llm_score(raw: str) -> float:
    s = re.search(r"<score>\s*\[?(\d+(?:\.\d+)?)\]?\s*</score>", raw or "")
    if not s:
        raise RuntimeError(f"No <score> tag:\n{raw}")
    return float(s.group(1))

# 4_2_2_{a_agent}_{a_model}__{b_agent}_{b_model}_scores_2.csv
PAIR_RE = re.compile(
    r"^4_2_2_(?P<a_agent>.+?)_(?P<a_model>.+?)__(?P<b_agent>.+?)_(?P<b_model>.+?)_scores_2\.csv$"
)

def parse_pair_from_4_2_2(name: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    m = PAIR_RE.match(name)
    if not m:
        raise ValueError(f"Bad pair filename: {name}")
    return (m.group("a_agent"), m.group("a_model")), (m.group("b_agent"), m.group("b_model"))

def converge_output_name(
    converger: Tuple[str, str],
    pair: Tuple[Tuple[str, str], Tuple[str, str]],
) -> str:
    # 4_2_{converger}_{A__B}_converge_2.csv（A__B 顺序来自 4_2_2 文件名）
    (ca, cm) = converger
    (a, b) = pair
    return f"4_2_{ca}_{cm}_{a[0]}_{a[1]}__{b[0]}_{b[1]}_converge_2.csv"

def step2_converge2_path(cfg: Config, converger: Tuple[str, str]) -> Path:
    # 2_{agent}_{model}_converge_2.csv
    return cfg.step2_dir / f"2_{cfg._safe(converger[0])}_{cfg._safe(converger[1])}_converge_2.csv"

def load_step2_comments(cfg: Config, converger: Tuple[str, str]) -> Dict[int, str]:
    """
    只取 step2 的评语（原样照搬）。
    返回：课程编号 -> 评语
    """
    p = step2_converge2_path(cfg, converger)
    if not p.exists():
        raise FileNotFoundError(f"Missing step2 converge_2 file for comments: {p}")

    df = read_csv_auto(p)
    need = {"课程编号", "评语"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{p} missing columns: {sorted(miss)}")

    out: Dict[int, str] = {}
    for _, r in df.iterrows():
        cid = int(r["课程编号"])
        out[cid] = str(r.get("评语", ""))
    return out

def load_step2_scores(cfg: Config, converger: Tuple[str, str]) -> Dict[int, float]:
    """
    取 step2 的分数（作为旧证据，供 converge_2 使用）。
    返回：课程编号 -> 分数
    """
    p = step2_converge2_path(cfg, converger)
    df = read_csv_auto(p)
    need = {"课程编号", "分数"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{p} missing columns: {sorted(miss)}")

    out: Dict[int, float] = {}
    for _, r in df.iterrows():
        cid = int(r["课程编号"])
        try:
            out[cid] = float(r["分数"])
        except Exception:
            continue
    return out

def build_mutual_evidence_line(r: pd.Series) -> str:
    """
    4_2_2 的互评数值证据：三项 A->B + 三项 B->A
    """
    return (
        "互评数值："
        f"A→B=[{r['a_score_standard_fit']},{r['a_score_consistency_fairness']},{r['a_score_accuracy_constructive']}]；"
        f"B→A=[{r['b_score_standard_fit']},{r['b_score_consistency_fairness']},{r['b_score_accuracy_constructive']}]；"
        f"final_score=[{r.get('final_score','')}]"
    )

# ============================================================
# 核心：单课 converge_2（生成新“分数”，评语=step2原样）
# ============================================================

def converge_one_lesson(
    lesson_id: int,
    converger: Tuple[str, str],
    pair_name: str,
    mutual_row: pd.Series,
    old_score: float,
    runner: ThreadRunner,
) -> Optional[Dict]:
    try:
        evidence = build_mutual_evidence_line(mutual_row)

        # 说明：评语不生成，后面直接用 step2 的评语原样写入
        scores_result = (
            f"pair_file: {pair_name}\n"
            f"converger: {converger[0]}_{converger[1]}\n\n"
            f"【旧证据：step2 converge_2 分数】\n"
            f"old_score=[{old_score}]\n\n"
            f"【新证据：4_2_2 mutual 数值证据】\n"
            f"{evidence}\n"
        )

        messages = [[
            {
                "role": "system",
                "content": converge_scoring_rubric["system"].format(criteria=RUBRIC_TEXT),
            },
            {
                "role": "user",
                "content": converge_scoring_rubric["user"].format(scores_result=scores_result),
            },
        ]]

        out = runner.run(messages)[0]
        score = parse_llm_score(out)

        return {
            "课程编号": int(lesson_id),
            "分数": float(score),
            # 评语：主流程里用 step2 原样覆盖（这里先留空，防止LLM乱写）
            "评语": "",
            "converger": f"{converger[0]}_{converger[1]}",
            "source_pair": pair_name,
        }

    except Exception:
        return None

# ============================================================
# 主流程：
# ✅ 只处理包含 --agent/--model 的 pair 文件
# ✅ 只由 --agent/--model 作为 converger 产出一份 converge_2
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)

    cfg = Config(
        agent_name=args.agent or Config.agent_name,
        model_name=args.model or Config.model_name,
        num_threads=args.num_threads or 30,
    )
    override_config_value(cfg, args, "result_dir", as_path=True)
    override_config_value(cfg, args, "step2_dir", as_path=True)

    cfg.result_dir.mkdir(parents=True, exist_ok=True)

    # 先扫描所有 mutual 文件
    all_pair_files = sorted(cfg.result_dir.glob("4_2_2_*_scores_2.csv"))
    if not all_pair_files:
        console.print(f"[bold red]No 4_2_2_*_scores_2.csv found in {cfg.result_dir}[/bold red]")
        return

    # ✅ 过滤：只保留文件名 pair 中包含当前执行模型的
    pair_files: List[Path] = []
    for p in all_pair_files:
        try:
            a, b = parse_pair_from_4_2_2(p.name)
        except ValueError:
            continue
        if (cfg.agent_name, cfg.model_name) in (a, b):
            pair_files.append(p)

    if not pair_files:
        console.print(
            f"[bold red]No 4_2_2 pair file matches --agent={cfg.agent_name} --model={cfg.model_name} in {cfg.result_dir}[/bold red]"
        )
        return

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)

    # ✅ converger 固定为执行模型
    converger: Tuple[str, str] = (cfg.agent_name, cfg.model_name)

    for pair_path in pair_files:
        pair_name = pair_path.name
        pair = parse_pair_from_4_2_2(pair_name)

        df_pair = read_csv_auto(pair_path)
        need_cols = {
            "课程编号",
            "final_score",
            "a_score_standard_fit",
            "a_score_consistency_fairness",
            "a_score_accuracy_constructive",
            "b_score_standard_fit",
            "b_score_consistency_fairness",
            "b_score_accuracy_constructive",
        }
        miss = need_cols - set(df_pair.columns)
        if miss:
            raise RuntimeError(f"{pair_path} missing columns: {sorted(miss)}")

        df_pair = df_pair.copy()
        df_pair["课程编号"] = df_pair["课程编号"].astype(int)
        df_pair_idx = df_pair.set_index("课程编号")
        all_lessons = sorted(df_pair["课程编号"].unique().tolist())

        # step2 的评语（完全照搬）& 旧分数（旧证据）
        comments_by_lesson = load_step2_comments(cfg, converger)
        old_score_by_lesson = load_step2_scores(cfg, converger)

        out_name = converge_output_name(converger, pair)
        out_path = cfg.result_dir / out_name

        # 断点续跑：按课程编号 done
        if out_path.exists():
            out_df = read_csv_auto(out_path)
            done: Set[int] = set(out_df["课程编号"].astype(int).tolist())
            result_df = out_df
        else:
            done = set()
            result_df = pd.DataFrame(columns=["课程编号", "分数", "评语", "converger", "source_pair"])

        pending = [
            cid for cid in all_lessons
            if cid not in done
            and cid in comments_by_lesson
            and cid in old_score_by_lesson
            and cid in df_pair_idx.index
        ]
        if not pending:
            console.print(f"[green]✅ {out_name} already done.[/green]")
            continue

        lock = threading.Lock()

        progress = build_score_progress()
        task_id = progress.add_task(
            "progress",
            total=len(all_lessons),
            completed=len(done),
        )

        with Live(progress, console=console, refresh_per_second=8):
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
                futures = {
                    exe.submit(
                        converge_one_lesson,
                        cid,
                        converger,
                        pair_name,
                        df_pair_idx.loc[cid],
                        float(old_score_by_lesson.get(cid, 0.0)),
                        runner,
                    ): cid
                    for cid in pending
                }

                for f in concurrent.futures.as_completed(futures):
                    cid = futures[f]
                    r = f.result()
                    if r:
                        # 评语严格用 step2 原样照搬
                        r["评语"] = comments_by_lesson.get(cid, "")
                        with lock:
                            result_df.loc[len(result_df)] = r
                            result_df.to_csv(out_path, index=False, encoding=cfg.output_encoding)
                    progress.advance(task_id)

        console.print(f"[bold green]✅ Saved => {out_path}[/bold green]")

if __name__ == "__main__":
    main()
