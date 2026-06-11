from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.live import Live

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_pool.runner import ThreadRunner
from evaluation.progress import build_score_progress
from evaluation.prompt_sets.essay import ESSAY_PROMPTS
from evaluation.essay_support import clean_generated_question_text, normalize_score_text, split_slug, stratified_sample_rows
from evaluation.question_bank import load_question_bank, load_question_dimensions
from evaluation.runtime import (
    add_path_override_argument,
    add_question_bank_argument,
    override_config_value,
    set_question_bank_override,
)


console = Console(force_terminal=True)

RESULT_COLUMNS = [
    "essay_id",
    "question_id",
    "question_trait",
    "question_text",
    "official_overall_score",
    "official_trait_score",
    "ai_score",
    "ai_comment",
]

ERROR_COLUMNS = [
    "essay_id",
    "question_id",
    "question_trait",
    "question_text",
    "error",
]

SCORE_VALUES = {"1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"}
DEFAULT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Essay key-problem scoring")
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=None)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-essays", type=int, default=None)
    parser.add_argument("--essay-id", action="append", dest="essay_ids", default=None)
    add_path_override_argument(parser, "--essay-file-path", "essay_file_path", "Override normalized essay CSV path.")
    add_path_override_argument(parser, "--score-file-path", "score_file_path", "Override essay score CSV path.")
    add_path_override_argument(parser, "--result-dir", "result_dir", "Override essay result directory root for this stage.")
    add_question_bank_argument(parser)
    return parser.parse_args()


@dataclass
class Config:
    essay_file_path: str = str(PROJECT_ROOT / "data" / "essay" / "origin" / "ellipse_train_normalized.csv")
    score_file_path: str = str(PROJECT_ROOT / "data" / "essay" / "score.csv")
    result_dir: str = str(PROJECT_ROOT / "data" / "essay" / "results")
    num_threads: int = 4
    split: str = DEFAULT_SPLIT
    tag: str = "2"
    agent_name: str = "qwen"
    model_name: str = "qwen3.5"
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def result_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"

    @property
    def error_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_errors.csv"


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def load_essay_rows(cfg: Config, score_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    essay_df = read_csv_auto(Path(cfg.essay_file_path))
    if cfg.split and split_slug(cfg.split) != "all":
        essay_df = essay_df[essay_df["split"].astype(str).str.lower() == cfg.split.lower()]
    if args.essay_ids:
        wanted = {str(item) for item in args.essay_ids}
        essay_df = essay_df[essay_df["essay_id"].astype(str).isin(wanted)]
    merged = essay_df.merge(score_df, on="essay_id", how="left", suffixes=("", "_score_ref")).reset_index(drop=True)
    if args.max_essays is not None:
        merged = stratified_sample_rows(merged, "overall_score", args.max_essays)
    return merged.reset_index(drop=True)


def parse_llm_response(raw: str) -> tuple[str, str]:
    score_match = re.search(r"<score>\s*([1-5](?:\\.0|\\.5)?)\s*</score>", raw, re.I | re.S)
    if not score_match:
        score_match = re.search(r"\b([1-5](?:\\.0|\\.5)?)\b", raw)
    if not score_match:
        raise ValueError(f"Cannot parse score from response:\n{raw}")
    score = normalize_score_text(score_match.group(1))
    if score not in SCORE_VALUES:
        raise ValueError(f"Unsupported score value: {score}")
    comment_match = re.search(r"<comment>\s*(.*?)\s*</comment>", raw, re.I | re.S)
    comment = comment_match.group(1).strip() if comment_match else raw.strip()
    return score, comment


def build_trait_reference_text(row: pd.Series, question_trait: str) -> str:
    official_value = row.get(question_trait, "")
    if pd.isna(official_value) or official_value == "":
        return ""
    return str(official_value)


def grade_single_problem(row: pd.Series, question_id: int, question_text: str, question_trait: str, runner: ThreadRunner) -> tuple[Optional[dict], Optional[dict]]:
    prompt = ESSAY_PROMPTS["essay_score"]
    messages = [[
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": prompt["user"].format(
                essay_text=str(row["full_text"]),
                prompt_name=str(row.get("prompt", "")),
                task_name=str(row.get("task", "")),
                grade=str(row.get("grade", "")),
                question_text=question_text,
                question_trait=question_trait,
            ),
        },
    ]]

    try:
        output = runner.run(messages)
        ai_score, ai_comment = parse_llm_response(output[0])
    except Exception as exc:
        return None, {
            "essay_id": str(row["essay_id"]),
            "question_id": question_id,
            "question_trait": question_trait,
            "question_text": question_text,
            "error": str(exc),
        }

    return {
        "essay_id": str(row["essay_id"]),
        "question_id": question_id,
        "question_trait": question_trait,
        "question_text": question_text,
        "official_overall_score": row.get("overall_score", ""),
        "official_trait_score": build_trait_reference_text(row, question_trait),
        "ai_score": ai_score,
        "ai_comment": ai_comment,
    }, None


def main() -> None:
    args = parse_args()
    set_question_bank_override(args)

    cfg = Config(
        agent_name=args.agent or Config.agent_name,
        model_name=args.model or Config.model_name,
        split=args.split,
    )
    override_config_value(cfg, args, "num_threads")
    override_config_value(cfg, args, "essay_file_path")
    override_config_value(cfg, args, "score_file_path")
    override_config_value(cfg, args, "result_dir")

    score_df = read_csv_auto(Path(cfg.score_file_path))
    essay_df = load_essay_rows(cfg, score_df, args)
    if essay_df.empty:
        print(f"Skipping essay key-problem scoring: no rows found for split={cfg.split}.")
        return

    question_bank = load_question_bank()
    if not question_bank:
        raise FileNotFoundError("Essay question bank is required before scoring.")

    question_dimensions = load_question_dimensions([])
    question_traits = question_bank.get("question_traits", [])
    if len(question_dimensions) != len(question_traits):
        raise ValueError("Essay question bank must contain question_traits aligned with question_dimensions.")

    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
    all_tasks = [(str(row["essay_id"]), qid) for _, row in essay_df.iterrows() for qid in range(1, len(question_dimensions) + 1)]

    if cfg.result_path.exists():
        result_df = read_csv_auto(cfg.result_path)
        done = {(str(record["essay_id"]), int(record["question_id"])) for _, record in result_df.iterrows()}
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    if cfg.error_path.exists():
        error_df = read_csv_auto(cfg.error_path)
    else:
        error_df = pd.DataFrame(columns=ERROR_COLUMNS)

    pending = [(essay_id, qid) for essay_id, qid in all_tasks if (essay_id, qid) not in done]
    if not pending:
        print(f"All essay key-problem tasks already completed: {cfg.result_path}")
        return

    row_lookup = {str(row["essay_id"]): row for _, row in essay_df.iterrows()}
    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    progress = build_score_progress()
    lock = threading.Lock()

    with Live(progress, console=console, refresh_per_second=8):
        task_id = progress.add_task("Essay key-problem scoring", total=len(pending))

        def worker(task: tuple[str, int]) -> tuple[Optional[dict], Optional[dict]]:
            essay_id, question_id = task
            row = row_lookup[essay_id]
            question_text = clean_generated_question_text(question_dimensions[question_id - 1])
            question_trait = question_traits[question_id - 1]
            result = grade_single_problem(row, question_id, question_text, question_trait, runner)
            with lock:
                progress.advance(task_id)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [executor.submit(worker, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()
                if result is not None:
                    result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)
                    result_df.to_csv(cfg.result_path, index=False, encoding=cfg.output_encoding)
                if error is not None:
                    error_df = pd.concat([error_df, pd.DataFrame([error])], ignore_index=True)
                    error_df.to_csv(cfg.error_path, index=False, encoding=cfg.output_encoding)

    success_count = len(result_df)
    error_count = len(error_df)
    if success_count == 0 and error_count > 0:
        raise RuntimeError(f"Essay key-problem scoring failed for all pending tasks. See: {cfg.error_path}")
    if error_count > 0:
        print(f"Essay key-problem results written to: {cfg.result_path}")
        print(f"Some tasks failed and were logged to: {cfg.error_path}")
        return
    print(f"Essay key-problem results written to: {cfg.result_path}")


if __name__ == "__main__":
    main()
