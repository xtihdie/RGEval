from __future__ import annotations

import argparse
import concurrent.futures
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.live import Live

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.progress import build_score_progress
from evaluation.prompt_sets.wiki import WIKI_PROMPTS
from evaluation.question_bank import load_question_bank
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override
from evaluation.wiki_quality_support import load_wiki_rows, parse_label_comment_response, split_slug
from llm_pool.runner import ThreadRunner


console = Console(force_terminal=True)

OVERALL_RESULT_COLUMNS = [
    "article_id",
    "official_label",
    "official_score",
    "question_evidence",
    "ai_label",
    "ai_score",
    "ai_comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Wikipedia key-question convergence")
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
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
    split: str = "train"
    tag: str = "2"
    agent_name: str = "deepseek"
    model_name: str = "deepseek-v3.2"
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace("/", "-")

    @property
    def question_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"wiki_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"

    @property
    def overall_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"wiki_{self.tag}_overall_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"


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

    wiki_df = load_wiki_rows(
        Path(cfg.metadata_file_path),
        split=cfg.split,
        article_ids=args.article_ids,
        max_articles=args.max_articles,
    )
    if wiki_df.empty:
        print(f"Skipping wiki key-question convergence: no rows found for split={cfg.split}.")
        return

    bank = load_question_bank()
    if not bank:
        raise FileNotFoundError("Wiki question bank is required for question convergence.")

    if not cfg.question_path.exists():
        raise FileNotFoundError(f"Wiki key-question result not found: {cfg.question_path}")
    question_df = pd.read_csv(cfg.question_path, encoding=cfg.output_encoding)

    if cfg.overall_path.exists():
        overall_df = pd.read_csv(cfg.overall_path, encoding=cfg.output_encoding)
        done = {str(value) for value in overall_df["article_id"].astype(str)}
    else:
        overall_df = pd.DataFrame(columns=OVERALL_RESULT_COLUMNS)
        done = set()

    pending_rows = [row for _, row in wiki_df.iterrows() if str(row["article_id"]) not in done]
    if not pending_rows:
        print(f"Wiki key-question convergence already completed: {cfg.overall_path}")
        return

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    prompt = WIKI_PROMPTS["wiki_overall_converge"]
    progress = build_score_progress()
    lock = threading.Lock()

    def worker(row: pd.Series) -> dict[str, object]:
        article_id = str(row["article_id"])
        subset = question_df[question_df["article_id"].astype(str) == article_id].sort_values("question_id")
        question_evidence = "\n".join(
            f"Q{record['question_id']} ({record['question_trait']}): label={record['ai_label']}; comment={record['ai_comment']}"
            for _, record in subset.iterrows()
        )
        messages = [[
            {"role": "system", "content": prompt["system"]},
            {
                "role": "user",
                "content": prompt["user"].format(
                    question_evidence=question_evidence,
                    official_label=str(row["quality_label"]),
                ),
            },
        ]]
        output = runner.run(messages)
        label, score, comment = parse_label_comment_response(output[0])
        return {
            "article_id": article_id,
            "official_label": str(row["quality_label"]),
            "official_score": row["quality_score"],
            "question_evidence": question_evidence,
            "ai_label": label,
            "ai_score": score,
            "ai_comment": comment,
        }

    with Live(progress, console=console, refresh_per_second=8):
        task_id = progress.add_task("Wiki key-question converge", total=len(pending_rows))
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [executor.submit(worker, row) for row in pending_rows]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    overall_df = pd.concat([overall_df, pd.DataFrame([result])], ignore_index=True)
                    overall_df.to_csv(cfg.overall_path, index=False, encoding=cfg.output_encoding)
                    progress.advance(task_id)

    print(f"Wiki key-question overall results written to: {cfg.overall_path}")


if __name__ == "__main__":
    main()
