from __future__ import annotations

import argparse
import concurrent.futures
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.live import Live

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from evaluation.essay_support import TRAIT_COLUMNS, load_essay_rows, parse_score_comment_response, split_slug
from evaluation.progress import build_score_progress
from evaluation.prompt_sets.essay import ESSAY_PROMPTS
from evaluation.question_bank import load_rubric_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override
from llm_pool.runner import ThreadRunner


console = Console(force_terminal=True)
DEFAULT_SPLIT = "train"

RESULT_COLUMNS = [
    "essay_id",
    "trait_id",
    "trait_name",
    "trait_description",
    "official_overall_score",
    "official_trait_score",
    "ai_score",
    "ai_comment",
]

ERROR_COLUMNS = ["essay_id", "trait_id", "trait_name", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Essay trait-level scoring")
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
    tag: str = "1"
    agent_name: str = "deepseek"
    model_name: str = "deepseek-v3.2"
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


def grade_trait(row: pd.Series, trait_index: int, trait_dimension: dict[str, str], runner: ThreadRunner) -> tuple[Optional[dict], Optional[dict]]:
    trait_name = trait_dimension["criteria"]
    prompt = ESSAY_PROMPTS["essay_trait_score"]
    messages = [[
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": prompt["user"].format(
                essay_text=str(row["full_text"]),
                prompt_name=str(row.get("prompt", "")),
                task_name=str(row.get("task", "")),
                grade=str(row.get("grade", "")),
                trait_name=trait_name,
                trait_description=trait_dimension.get("expound", ""),
            ),
        },
    ]]
    try:
        output = runner.run(messages)
        ai_score, ai_comment = parse_score_comment_response(output[0])
    except Exception as exc:
        return None, {
            "essay_id": str(row["essay_id"]),
            "trait_id": trait_index,
            "trait_name": trait_name,
            "error": str(exc),
        }

    official_trait_score = row.get(TRAIT_COLUMNS[trait_index - 1], "")
    return {
        "essay_id": str(row["essay_id"]),
        "trait_id": trait_index,
        "trait_name": trait_name,
        "trait_description": trait_dimension.get("expound", ""),
        "official_overall_score": row.get("overall_score", ""),
        "official_trait_score": official_trait_score,
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

    rubric_dimensions = load_rubric_dimensions([])
    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
    essay_df = load_essay_rows(
        Path(cfg.essay_file_path),
        Path(cfg.score_file_path),
        split=cfg.split,
        essay_ids=args.essay_ids,
        max_essays=args.max_essays,
    )

    if cfg.result_path.exists():
        result_df = pd.read_csv(cfg.result_path, encoding=cfg.output_encoding)
        done = {(str(r["essay_id"]), int(r["trait_id"])) for _, r in result_df.iterrows()}
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()
    if cfg.error_path.exists():
        error_df = pd.read_csv(cfg.error_path, encoding=cfg.output_encoding)
    else:
        error_df = pd.DataFrame(columns=ERROR_COLUMNS)

    pending = []
    for _, row in essay_df.iterrows():
        for index, trait_dimension in enumerate(rubric_dimensions, start=1):
            key = (str(row["essay_id"]), index)
            if key not in done:
                pending.append((row, index, trait_dimension))
    if not pending:
        print(f"All essay trait-scoring tasks already completed: {cfg.result_path}")
        return

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    progress = build_score_progress()
    lock = threading.Lock()
    with Live(progress, console=console, refresh_per_second=8):
        task_id = progress.add_task("Essay trait scoring", total=len(pending))

        def worker(item: tuple[pd.Series, int, dict[str, str]]) -> tuple[Optional[dict], Optional[dict]]:
            row, trait_index, trait_dimension = item
            result = grade_trait(row, trait_index, trait_dimension, runner)
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

    if len(result_df) == 0 and len(error_df) > 0:
        raise RuntimeError(f"Essay trait scoring failed for all pending tasks. See: {cfg.error_path}")
    print(f"Essay trait-scoring results written to: {cfg.result_path}")


if __name__ == "__main__":
    main()
