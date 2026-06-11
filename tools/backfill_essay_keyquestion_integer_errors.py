from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.essay_support import normalize_score_text, read_csv_auto


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Backfill essay key-question results from integer-score error logs")
    parser.add_argument("--score-file-path", default=str(PROJECT_ROOT / "data" / "essay" / "score.csv"))
    parser.add_argument("--result-dir", default=str(PROJECT_ROOT / "data" / "essay" / "results" / "2"))
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace("/", "-")


def main() -> None:
    args = parse_args()
    agent = safe_name(args.agent)
    model = safe_name(args.model)
    result_dir = Path(args.result_dir)
    error_path = result_dir / f"essay_2_{args.split}_{agent}_{model}_errors.csv"
    result_path = result_dir / f"essay_2_{args.split}_{agent}_{model}_scores.csv"

    if not error_path.exists():
        raise FileNotFoundError(f"Error file not found: {error_path}")

    error_df = read_csv_auto(error_path)
    if error_df.empty:
        print(f"No rows found in: {error_path}")
        return

    score_df = read_csv_auto(Path(args.score_file_path))
    score_df["essay_id"] = score_df["essay_id"].astype(str)

    recovered_mask = error_df["error"].astype(str).str.match(r"^Unsupported score value:\s*[1-5]\s*$")
    recovered_df = error_df[recovered_mask].copy()
    remaining_df = error_df[~recovered_mask].copy()

    if recovered_df.empty:
        print(f"No integer-score rows found to recover in: {error_path}")
        return

    recovered_df["essay_id"] = recovered_df["essay_id"].astype(str)
    recovered_df["question_id"] = recovered_df["question_id"].astype(int)
    recovered_df["ai_score"] = recovered_df["error"].astype(str).str.extract(r"([1-5])", expand=False).map(normalize_score_text)
    recovered_df["ai_comment"] = ""

    merged = recovered_df.merge(score_df, on="essay_id", how="left", suffixes=("", "_score_ref"))
    merged["official_overall_score"] = merged["overall_score"]
    merged["official_trait_score"] = merged.apply(
        lambda row: row.get(str(row["question_trait"]).strip().lower(), ""),
        axis=1,
    )

    recovered_results = merged[
        [
            "essay_id",
            "question_id",
            "question_trait",
            "question_text",
            "official_overall_score",
            "official_trait_score",
            "ai_score",
            "ai_comment",
        ]
    ].copy()

    if result_path.exists():
        result_df = read_csv_auto(result_path)
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)

    combined_df = pd.concat([result_df, recovered_results], ignore_index=True)
    combined_df["essay_id"] = combined_df["essay_id"].astype(str)
    combined_df["question_id"] = combined_df["question_id"].astype(int)
    combined_df = combined_df.drop_duplicates(subset=["essay_id", "question_id"], keep="first")
    combined_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    remaining_df.to_csv(error_path, index=False, encoding="utf-8-sig")
    print(f"Recovered {len(recovered_results)} rows into: {result_path}")
    print(f"Remaining error rows kept in: {error_path}")


if __name__ == "__main__":
    main()
