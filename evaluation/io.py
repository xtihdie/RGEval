from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


COMMON_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin1")


def read_csv_auto(path: str | Path, encodings: Iterable[str] = COMMON_ENCODINGS) -> pd.DataFrame:
    csv_path = Path(path)
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - depends on local files
            last_error = exc
    raise ValueError(f"Unable to read CSV with known encodings: {csv_path}") from last_error


def discover_numeric_prefixes(root: str | Path, suffix: str = ".csv") -> list[int]:
    base_dir = Path(root)
    values: list[int] = []
    for item in base_dir.iterdir():
        if item.is_file() and item.name.endswith(suffix):
            prefix = item.name.split("_", 1)[0]
            if prefix.isdigit():
                values.append(int(prefix))
    return sorted(values)


def find_first_matching_csv(root: str | Path, record_id: int) -> Path | None:
    base_dir = Path(root)
    prefix = str(record_id)
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv" and item.name.startswith(prefix):
            return item
    return None


def sanitize_name(value: str) -> str:
    return value.replace("/", "-")
