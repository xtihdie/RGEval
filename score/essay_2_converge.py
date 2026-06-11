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

from evaluation.essay_support import build_all_trait_reference_text, load_essay_rows, parse_score_comment_response, split_slug
from evaluation.progress import build_score_progress
from evaluation.prompt_sets.essay import ESSAY_PROMPTS
from evaluation.question_bank import load_question_bank
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override
from llm_pool.runner import ThreadRunner


console = Console(force_terminal=True)

TRAIT_RESULT_COLUMNS = [
    "essay_id",
    "trait_id",
    "trait_name",
    "question_evidence",
    "official_trait_score",
    "ai_score",
    "ai_comment",
]

OVERALL_RESULT_COLUMNS = [
    "essay_id",
    "official_overall_score",
    "trait_evidence",
    "ai_score",
    "ai_comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Essay key-question convergence")
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
    tag: str = "2"
    agent_name: str = "deepseek"
    model_name: str = "deepseek-v3.2"
    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def question_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"

    @property
    def trait_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_trait_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"

    @property
    def overall_path(self) -> Path:
        split = split_slug(self.split)
        return Path(self.result_dir) / self.tag / f"essay_{self.tag}_overall_{split}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"


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

    essay_df = load_essay_rows(
        Path(cfg.essay_file_path),
        Path(cfg.score_file_path),
        split=cfg.split,
        essay_ids=args.essay_ids,
        max_essays=args.max_essays,
    )
    if essay_df.empty:
        print(f"Skipping essay key-question convergence: no rows found for split={cfg.split}.")
        return

    bank = load_question_bank()
    if not bank:
        raise FileNotFoundError("Essay question bank is required for question convergence.")
    question_traits = bank.get("question_traits", [])
    trait_question_groups = bank.get("trait_question_groups", [])
    trait_names = [value["criteria"] for value in bank.get("rubric_dimensions", [])]
    if not question_traits or not trait_question_groups or not trait_names:
        raise ValueError("Essay question bank must contain question_traits, trait_question_groups, and rubric_dimensions.")

    if not cfg.question_path.exists():
        raise FileNotFoundError(f"Key-question result not found: {cfg.question_path}")
    question_df = pd.read_csv(cfg.question_path, encoding=cfg.output_encoding)

    if cfg.trait_path.exists():
        trait_df = pd.read_csv(cfg.trait_path, encoding=cfg.output_encoding)
    else:
        trait_df = pd.DataFrame(columns=TRAIT_RESULT_COLUMNS)
    if cfg.overall_path.exists():
        overall_df = pd.read_csv(cfg.overall_path, encoding=cfg.output_encoding)
        done_overall = {str(value) for value in overall_df["essay_id"].astype(str)}
    else:
        overall_df = pd.DataFrame(columns=OVERALL_RESULT_COLUMNS)
        done_overall = set()

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    trait_prompt = ESSAY_PROMPTS["essay_question_to_trait_converge"]
    overall_prompt = ESSAY_PROMPTS["essay_overall_converge"]
    pending_rows = []
    for _, row in essay_df.iterrows():
        essay_id = str(row["essay_id"])
        need_trait = True
        if not trait_df.empty:
            existing_trait_ids = set(trait_df[trait_df["essay_id"].astype(str) == essay_id]["trait_id"].astype(int).tolist())
            need_trait = len(existing_trait_ids) < len(trait_names)
        need_overall = essay_id not in done_overall
        if need_trait or need_overall:
            pending_rows.append(row)

    if not pending_rows:
        print(f"Essay key-question convergence already completed: {cfg.overall_path}")
        return

    progress = build_score_progress()
    lock = threading.Lock()

    def worker(row: pd.Series) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        essay_id = str(row["essay_id"])
        essay_questions = question_df[question_df["essay_id"].astype(str) == essay_id]
        trait_results: list[dict[str, object]] = []
        existing_trait_ids = set()
        if not trait_df.empty:
            existing_trait_ids = set(trait_df[trait_df["essay_id"].astype(str) == essay_id]["trait_id"].astype(int).tolist())

        for trait_id, (trait_name, question_ids) in enumerate(zip(trait_names, trait_question_groups), start=1):
            if trait_id in existing_trait_ids:
                continue
            subset = essay_questions[essay_questions["question_id"].isin(question_ids)].sort_values("question_id")
            question_evidence = "\n".join(
                f"Q{record['question_id']} ({record['question_text']}): score={record['ai_score']}; comment={record['ai_comment']}"
                for _, record in subset.iterrows()
            )
            messages = [[
                {"role": "system", "content": trait_prompt["system"]},
                {
                    "role": "user",
                    "content": trait_prompt["user"].format(
                        trait_name=trait_name,
                        question_evidence=question_evidence,
                    ),
                },
            ]]
            output = runner.run(messages)
            ai_score, ai_comment = parse_score_comment_response(output[0])
            trait_results.append(
                {
                    "essay_id": essay_id,
                    "trait_id": trait_id,
                    "trait_name": trait_name,
                    "question_evidence": question_evidence,
                    "official_trait_score": row.get(trait_name.lower(), ""),
                    "ai_score": ai_score,
                    "ai_comment": ai_comment,
                }
            )

        if essay_id in done_overall:
            return trait_results, None

        combined_trait_df = trait_df[trait_df["essay_id"].astype(str) == essay_id].copy() if not trait_df.empty else pd.DataFrame(columns=TRAIT_RESULT_COLUMNS)
        if trait_results:
            combined_trait_df = pd.concat([combined_trait_df, pd.DataFrame(trait_results)], ignore_index=True)
        combined_trait_df = combined_trait_df.sort_values("trait_id")
        trait_evidence = "\n".join(
            f"{record['trait_name']}: score={record['ai_score']}; comment={record['ai_comment']}"
            for _, record in combined_trait_df.iterrows()
        )
        messages = [[
            {"role": "system", "content": overall_prompt["system"]},
            {
                "role": "user",
                "content": overall_prompt["user"].format(
                    trait_evidence=trait_evidence,
                    official_trait_scores=build_all_trait_reference_text(row),
                ),
            },
        ]]
        output = runner.run(messages)
        ai_score, ai_comment = parse_score_comment_response(output[0])
        overall_result = {
            "essay_id": essay_id,
            "official_overall_score": row.get("overall_score", ""),
            "trait_evidence": trait_evidence,
            "ai_score": ai_score,
            "ai_comment": ai_comment,
        }
        return trait_results, overall_result

    with Live(progress, console=console, refresh_per_second=8):
        task_id = progress.add_task("Essay key-question converge", total=len(pending_rows))
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [executor.submit(worker, row) for _, row in enumerate(pending_rows)]
            for future in concurrent.futures.as_completed(futures):
                trait_results, overall_result = future.result()
                with lock:
                    if trait_results:
                        trait_df = pd.concat([trait_df, pd.DataFrame(trait_results)], ignore_index=True)
                        trait_df.to_csv(cfg.trait_path, index=False, encoding=cfg.output_encoding)
                    if overall_result is not None:
                        overall_df = pd.concat([overall_df, pd.DataFrame([overall_result])], ignore_index=True)
                        overall_df.to_csv(cfg.overall_path, index=False, encoding=cfg.output_encoding)
                    progress.advance(task_id)

    print(f"Essay key-question trait results written to: {cfg.trait_path}")
    print(f"Essay key-question overall results written to: {cfg.overall_path}")


if __name__ == "__main__":
    main()
