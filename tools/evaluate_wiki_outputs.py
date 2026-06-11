from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.wiki_metrics import compute_label_metrics
from evaluation.wiki_quality_support import split_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate wiki experiment outputs")
    parser.add_argument("--result-dir", default=str(PROJECT_ROOT / "data" / "wiki_quality" / "results"))
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace("/", "-")


def evaluate_table(df: pd.DataFrame, truth_col: str, pred_col: str, method: str, split: str, level: str) -> dict[str, object]:
    metrics = compute_label_metrics(df, truth_col=truth_col, pred_col=pred_col)
    metrics["method"] = method
    metrics["split"] = split
    metrics["level"] = level
    return metrics


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    split = split_slug(args.split)
    agent = safe_name(args.agent)
    model = safe_name(args.model)

    rows: list[dict[str, object]] = []

    direct_path = result_dir / "0" / f"wiki_0_{split}_{agent}_{model}_scores.csv"
    if direct_path.exists():
        direct_df = pd.read_csv(direct_path, encoding="utf-8-sig")
        rows.append(evaluate_table(direct_df, "official_label", "ai_label", "direct", split, "overall"))

    keyquestion_path = result_dir / "2" / f"wiki_2_{split}_{agent}_{model}_scores.csv"
    if keyquestion_path.exists():
        key_df = pd.read_csv(keyquestion_path, encoding="utf-8-sig")
        rows.append(evaluate_table(key_df, "official_label", "ai_label", "keyquestion", split, "question_all"))
        for trait_name in sorted(key_df["question_trait"].dropna().astype(str).unique().tolist()):
            subset = key_df[key_df["question_trait"].astype(str) == trait_name]
            if not subset.empty:
                rows.append(evaluate_table(subset, "official_label", "ai_label", "keyquestion", split, f"question_{trait_name}"))

    keyquestion_overall_path = result_dir / "2" / f"wiki_2_overall_{split}_{agent}_{model}_scores.csv"
    if keyquestion_overall_path.exists():
        key_overall_df = pd.read_csv(keyquestion_overall_path, encoding="utf-8-sig")
        rows.append(evaluate_table(key_overall_df, "official_label", "ai_label", "keyquestion", split, "overall"))

    if not rows:
        raise FileNotFoundError(f"No result files found for split={split}, agent={agent}, model={model}")

    metrics_df = pd.DataFrame(rows)
    output_path = result_dir / "metrics" / f"wiki_metrics_{split}_{agent}_{model}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
