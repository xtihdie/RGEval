from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


WIKI_LABELS = ["Stub", "Start", "C", "B", "GA", "FA"]
WIKI_LABEL_TO_SCORE = {label: index for index, label in enumerate(WIKI_LABELS)}
WIKI_SCORE_TO_LABEL = {index: label for label, index in WIKI_LABEL_TO_SCORE.items()}


def split_slug(split: str | None) -> str:
    if not split:
        return "all"
    lowered = str(split).strip().lower()
    return lowered or "all"


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def load_wiki_rows(
    metadata_path: Path,
    *,
    split: str,
    article_ids: list[str] | None = None,
    max_articles: int | None = None,
) -> pd.DataFrame:
    df = read_csv_auto(metadata_path)
    if split_slug(split) != "all":
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    if article_ids:
        wanted = {str(value) for value in article_ids}
        df = df[df["article_id"].astype(str).isin(wanted)]
    df = df[df["text_exists"].astype(str).str.lower() == "true"].reset_index(drop=True)
    if max_articles is not None:
        if len(df) > max_articles:
            groups = [group.copy() for _, group in df.groupby("quality_label", sort=True)]
            per_group = max_articles // len(groups)
            remainder = max_articles % len(groups)
            target_counts = []
            for index, group in enumerate(groups):
                target_counts.append(min(len(group), per_group + (1 if index < remainder else 0)))
            remaining = max_articles - sum(target_counts)
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
            sampled: list[pd.DataFrame] = []
            for index, group in enumerate(groups):
                take = target_counts[index]
                if take > 0:
                    sampled.append(group.head(take))
            df = pd.concat(sampled, ignore_index=True) if sampled else df.head(0)
    return df


def load_article_text(text_path: str | Path) -> str:
    path = Path(text_path)
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_label(value: str) -> str:
    text = str(value).strip()
    normalized = re.sub(r"[\s_-]+", " ", text).strip().upper()
    if normalized in {"STUB", "STUB CLASS"}:
        return "Stub"
    if normalized in {"START", "START CLASS"}:
        return "Start"
    if normalized in {"C", "C CLASS"}:
        return "C"
    if normalized in {"B", "B CLASS"}:
        return "B"
    if normalized in {"GA", "GOOD ARTICLE"}:
        return "GA"
    if normalized in {"FA", "FEATURED ARTICLE"}:
        return "FA"
    raise ValueError(f"Unsupported Wikipedia quality label: {value}")


def parse_label_comment_response(raw: str) -> tuple[str, int, str]:
    label = None
    label_match = re.search(r"<label>\s*(.*?)\s*</label>", raw, re.I | re.S)
    if label_match:
        candidate = label_match.group(1).strip()
        if candidate and candidate.lower() not in {"quality_class", "quality class", "class"}:
            try:
                label = normalize_label(candidate)
            except ValueError:
                label = None

    if label is None:
        free_match = re.search(
            r"\b(Featured Article|Good Article|Stub(?:-Class)?|Start(?:-Class)?|C(?:-Class)?|B(?:-Class)?|GA|FA)\b",
            raw,
            re.I,
        )
        if not free_match:
            raise ValueError(f"Cannot parse Wikipedia quality label from response:\n{raw}")
        label = normalize_label(free_match.group(1))
    comment_match = re.search(r"<comment>\s*(.*?)\s*</comment>", raw, re.I | re.S)
    comment = comment_match.group(1).strip() if comment_match else raw.strip()
    return label, WIKI_LABEL_TO_SCORE[label], comment
