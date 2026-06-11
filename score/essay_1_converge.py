from __future__ import annotations

import argparse
import concurrent.futures
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.live import Live

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from evaluation.essay_support import build_all_trait_reference_text, load_essay_rows, parse_score_comment_response, split_slug
from evaluation.progress import build_score_progress
from evaluation.prompt_sets.essay import ESSAY_PROMPTS
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value
from llm_pool.runner import ThreadRunner


console = Console(force_terminal=True)

RESULT_COLUMNS = [
    "essay_id",
    "official_overall_score",
    "trait_evidence",
    "ai_score",
    "ai_comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Essay trait-to-overall convergence")
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
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
    split: str = "train"
    tag: str = "1"
    agent_name: str = "deepseek"
    model_name: str = "deepseek-v3.2"
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def trait_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"

    @property
    def overall_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_overall_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"


def main() -> None:
    args = parse_args()
    cfg = Config(
        agent_name=args.agent or Config.agent_name,
        model_name=args.model or Config.model_name,
        split=args.split,
    )
    override_config_value(cfg, args, "num_threads")
    override_config_value(cfg, args, "essay_file_path")
    override_config_value(cfg, args, "score_file_path")
    override_config_value(cfg, args, "result_dir")

    if not cfg.trait_path.exists():
        raise FileNotFoundError(f"Trait-scoring result not found: {cfg.trait_path}")

    essay_df = load_essay_rows(
        Path(cfg.essay_file_path),
        Path(cfg.score_file_path),
        split=cfg.split,
        essay_ids=args.essay_ids,
        max_essays=args.max_essays,
    )
    trait_df = pd.read_csv(cfg.trait_path, encoding=cfg.output_encoding)
    if cfg.overall_path.exists():
        overall_df = pd.read_csv(cfg.overall_path, encoding=cfg.output_encoding)
        done = {str(value) for value in overall_df["essay_id"].astype(str)}
    else:
        overall_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    prompt = ESSAY_PROMPTS["essay_overall_converge"]
    pending_rows = [row for _, row in essay_df.iterrows() if str(row["essay_id"]) not in done]
    if not pending_rows:
        print(f"Essay trait-converged overall results already completed: {cfg.overall_path}")
        return

    progress = build_score_progress()
    lock = threading.Lock()

    def worker(row: pd.Series) -> dict[str, object]:
        essay_id = str(row["essay_id"])
        rows = trait_df[trait_df["essay_id"].astype(str) == essay_id].sort_values("trait_id")
        trait_evidence = "\n".join(
            f"{record['trait_name']}: score={record['ai_score']}; comment={record['ai_comment']}"
            for _, record in rows.iterrows()
        )
        messages = [[
            {"role": "system", "content": prompt["system"]},
            {
                "role": "user",
                "content": prompt["user"].format(
                    trait_evidence=trait_evidence,
                    official_trait_scores=build_all_trait_reference_text(row),
                ),
            },
        ]]
        output = runner.run(messages)
        ai_score, ai_comment = parse_score_comment_response(output[0])
        return {
            "essay_id": essay_id,
            "official_overall_score": row.get("overall_score", ""),
            "trait_evidence": trait_evidence,
            "ai_score": ai_score,
            "ai_comment": ai_comment,
        }

    with Live(progress, console=console, refresh_per_second=8):
        task_id = progress.add_task("Essay trait converge", total=len(pending_rows))
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [executor.submit(worker, row) for row in pending_rows]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    overall_df = pd.concat([overall_df, pd.DataFrame([result])], ignore_index=True)
                    overall_df.to_csv(cfg.overall_path, index=False, encoding=cfg.output_encoding)
                    progress.advance(task_id)

    print(f"Essay trait-converged overall results written to: {cfg.overall_path}")


if __name__ == "__main__":
    main()
