from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TRAIT_COLUMNS = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]

TRAIT_DISPLAY = {
    "cohesion": "Cohesion",
    "syntax": "Syntax",
    "vocabulary": "Vocabulary",
    "phraseology": "Phraseology",
    "grammar": "Grammar",
    "conventions": "Conventions",
}

SCORE_VALUES = {"1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"}


def stratified_sample_rows(df: pd.DataFrame, label_col: str, sample_size: int) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df.reset_index(drop=True)
    groups = [group.copy() for _, group in df.groupby(label_col, sort=True)]
    if not groups:
        return df.head(0).reset_index(drop=True)
    target_counts = []
    per_group = sample_size // len(groups)
    remainder = sample_size % len(groups)
    for index, group in enumerate(groups):
        target_counts.append(min(len(group), per_group + (1 if index < remainder else 0)))

    remaining = sample_size - sum(target_counts)
    while remaining > 0:
        progressed = False
        for index, group in enumerate(groups):
            if target_counts[index] < len(group):
                target_counts[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    sampled = [group.head(target_counts[index]) for index, group in enumerate(groups) if target_counts[index] > 0]
    if not sampled:
        return df.head(0).reset_index(drop=True)
    return pd.concat(sampled, ignore_index=True).reset_index(drop=True)


def normalize_score_text(score: str) -> str:
    value = str(score).strip()
    if re.fullmatch(r"[1-5]", value):
        return f"{value}.0"
    return value


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def load_essay_rows(
    essay_file_path: Path,
    score_file_path: Path,
    *,
    split: str | None = None,
    essay_ids: list[str] | None = None,
    max_essays: int | None = None,
) -> pd.DataFrame:
    essay_df = read_csv_auto(essay_file_path)
    score_df = read_csv_auto(score_file_path)
    essay_df["essay_id"] = essay_df["essay_id"].astype(str)
    score_df["essay_id"] = score_df["essay_id"].astype(str)
    if split and split_slug(split) != "all":
        essay_df = essay_df[essay_df["split"].astype(str).str.lower() == split.lower()]
    if essay_ids:
        wanted = {str(item) for item in essay_ids}
        essay_df = essay_df[essay_df["essay_id"].isin(wanted)]
    merged = essay_df.merge(score_df, on="essay_id", how="left", suffixes=("", "_score_ref")).reset_index(drop=True)
    if max_essays is not None:
        merged = stratified_sample_rows(merged, "overall_score", max_essays)
    return merged.reset_index(drop=True)


def parse_score_comment_response(raw: str) -> tuple[str, str]:
    score_match = re.search(r"<score>\s*([1-5](?:\.0|\.5)?)\s*</score>", raw, re.I | re.S)
    if not score_match:
        score_match = re.search(r"\b([1-5](?:\.0|\.5)?)\b", raw)
    if not score_match:
        raise ValueError(f"Cannot parse score from response:\n{raw}")
    score = normalize_score_text(score_match.group(1))
    if score not in SCORE_VALUES:
        raise ValueError(f"Unsupported score value: {score}")
    comment_match = re.search(r"<comment>\s*(.*?)\s*</comment>", raw, re.I | re.S)
    comment = comment_match.group(1).strip() if comment_match else raw.strip()
    return score, comment


def build_all_trait_reference_text(row: pd.Series) -> str:
    parts = []
    for trait in TRAIT_COLUMNS:
        value = row.get(trait, "")
        if pd.notna(value) and value != "":
            parts.append(f"{trait}={value}")
    return ", ".join(parts)


def clean_generated_question_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("é”›?", "?").replace("？", "?")
    cleaned = re.sub(r"[^\x00-\x7F?]+$", "", cleaned).strip()
    if cleaned and not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".") + "?"
    return cleaned


def split_slug(split: str | None) -> str:
    if not split:
        return "all"
    value = str(split).strip().lower()
    if value in {"", "all", "*"}:
        return "all"
    return value
