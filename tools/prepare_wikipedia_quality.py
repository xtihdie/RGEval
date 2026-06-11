from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "wiki_quality"
ORIGIN_ROOT = DATASET_ROOT / "origin"
DATASETS_DIR = ORIGIN_ROOT / "datasets"
REVISION_DIR = ORIGIN_ROOT / "revisiondata"
OUTPUT_PATH = DATASET_ROOT / "metadata.csv"

RATING_TO_SCORE = {
    "Stub": 0,
    "Start": 1,
    "C": 2,
    "B": 3,
    "GA": 4,
    "FA": 5,
}


def load_split(name: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(DATASETS_DIR / name, sep="\t")
    df["split"] = split
    return df


def build_metadata() -> pd.DataFrame:
    train_df = load_split("training-set.tsv", "train")
    test_df = load_split("test-set.tsv", "test")
    df = pd.concat([train_df, test_df], ignore_index=True)

    df["article_pageid"] = df["article_pageid"].astype(str)
    df["article_revid"] = df["article_revid"].astype(str)
    df["talk_pageid"] = df["talk_pageid"].astype(str)
    df["talk_revid"] = df["talk_revid"].astype(str)
    df["quality_label"] = df["rating"].astype(str)
    df["quality_score"] = df["quality_label"].map(RATING_TO_SCORE)
    df["text_path"] = df["article_revid"].map(lambda value: str((REVISION_DIR / value).resolve()))
    df["text_exists"] = df["article_revid"].map(lambda value: (REVISION_DIR / value).exists())
    df["article_id"] = df["article_revid"]

    columns = [
        "article_id",
        "article_pageid",
        "article_revid",
        "talk_pageid",
        "talk_revid",
        "split",
        "quality_label",
        "quality_score",
        "text_path",
        "text_exists",
    ]
    return df[columns].sort_values(["split", "quality_score", "article_id"]).reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_metadata()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    missing = int((~df["text_exists"]).sum())
    print(f"Metadata written to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Train rows: {(df['split'] == 'train').sum()}")
    print(f"Test rows: {(df['split'] == 'test').sum()}")
    print(f"Missing revision texts: {missing}")
    print(df["quality_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
