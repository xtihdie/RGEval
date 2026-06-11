from __future__ import annotations

import argparse
import concurrent.futures
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

from evaluation.progress import build_score_progress
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override
from evaluation.wiki_quality_support import load_article_text, load_wiki_rows, parse_label_comment_response, split_slug
from evaluation.prompt_sets.wiki import WIKI_PROMPTS
from llm_pool.runner import ThreadRunner


console = Console(force_terminal=True)

RESULT_COLUMNS = [
    "article_id",
    "split",
    "official_label",
    "official_score",
    "ai_label",
    "ai_score",
    "ai_comment",
]

ERROR_COLUMNS = ["article_id", "error"]
DEFAULT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Wikipedia direct overall scoring")
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=None)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--article-id", action="append", dest="article_ids", default=None)
    add_path_override_argument(parser, "--metadata-file-path", "metadata_file_path", "Override wiki metadata CSV path.")
    add_path_override_argument(parser, "--result-dir", "result_dir", "Override wiki result directory root for this stage.")
    add_question_bank_argument(parser)
    return parser.parse_args()


@dataclass
class Config:
    metadata_file_path: str = str(PROJECT_ROOT / "data" / "wiki_quality" / "metadata.csv")
    result_dir: str = str(PROJECT_ROOT / "data" / "wiki_quality" / "results")
    num_threads: int = 4
    split: str = DEFAULT_SPLIT
    tag: str = "0"
    agent_name: str = "deepseek"
    model_name: str = "deepseek-v3.2"
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace("/", "-")

    @property
    def result_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"wiki_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"

    @property
    def error_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"wiki_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_errors.csv"


def grade_single_article(row: pd.Series, runner: ThreadRunner) -> tuple[Optional[dict], Optional[dict]]:
    prompt = WIKI_PROMPTS["wiki_overall_score"]
    article_text = load_article_text(row["text_path"])
    messages = [[
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(article_text=article_text)},
    ]]
    try:
        output = runner.run(messages)
        label, score, comment = parse_label_comment_response(output[0])
    except Exception as exc:
        return None, {"article_id": str(row["article_id"]), "error": str(exc)}
    return {
        "article_id": str(row["article_id"]),
        "split": str(row["split"]),
        "official_label": str(row["quality_label"]),
        "official_score": row["quality_score"],
        "ai_label": label,
        "ai_score": score,
        "ai_comment": comment,
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
    override_config_value(cfg, args, "metadata_file_path")
    override_config_value(cfg, args, "result_dir")

    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
    wiki_df = load_wiki_rows(
        Path(cfg.metadata_file_path),
        split=cfg.split,
        article_ids=args.article_ids,
        max_articles=args.max_articles,
    )
    if wiki_df.empty:
        print(f"Skipping wiki direct scoring: no rows found for split={cfg.split}.")
        return

    if cfg.result_path.exists():
        result_df = pd.read_csv(cfg.result_path, encoding=cfg.output_encoding)
        done = {str(value) for value in result_df["article_id"].astype(str)}
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    if cfg.error_path.exists():
        error_df = pd.read_csv(cfg.error_path, encoding=cfg.output_encoding)
    else:
        error_df = pd.DataFrame(columns=ERROR_COLUMNS)

    pending_rows = [row for _, row in wiki_df.iterrows() if str(row["article_id"]) not in done]
    if not pending_rows:
        print(f"All wiki direct-scoring tasks already completed: {cfg.result_path}")
        return

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    progress = build_score_progress()
    lock = threading.Lock()
    with Live(progress, console=console, refresh_per_second=8):
        task_id = progress.add_task("Wiki direct scoring", total=len(pending_rows))

        def worker(row: pd.Series) -> tuple[Optional[dict], Optional[dict]]:
            result = grade_single_article(row, runner)
            with lock:
                progress.advance(task_id)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [executor.submit(worker, row) for row in pending_rows]
            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()
                if result is not None:
                    result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)
                    result_df.to_csv(cfg.result_path, index=False, encoding=cfg.output_encoding)
                if error is not None:
                    error_df = pd.concat([error_df, pd.DataFrame([error])], ignore_index=True)
                    error_df.to_csv(cfg.error_path, index=False, encoding=cfg.output_encoding)

    if len(result_df) == 0 and len(error_df) > 0:
        raise RuntimeError(f"Wiki direct scoring failed for all pending tasks. See: {cfg.error_path}")
    print(f"Wiki direct-scoring results written to: {cfg.result_path}")


if __name__ == "__main__":
    main()
