from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import DATA_DIR
from .runtime import QUESTION_BANK_ENV


DEFAULT_QUESTION_BANK_PATH = DATA_DIR / "300class" / "question_bank.json"
DEFAULT_HARDCODED_BACKUP_PATH = DATA_DIR / "300class" / "question_bank_hardcoded_backup.json"

LEGACY_BACKUP_KEY_PATHS: dict[str, list[tuple[str, str]]] = {
    "rubric_text_full": [
        ("score/0_score.py", "RUBRIC"),
        ("score/4_0_mutual_score.py", "RUBRIC"),
        ("score/4_1_2_mutual_score.py", "RUBRIC"),
    ],
    "rubric_text_compact": [
        ("score/4_2_3_mutual_score.py", "RUBRIC"),
    ],
    "rubric_dimensions": [
        ("score/1_score.py", "RUBRIC_ARRAY"),
        ("score/1_converge.py", "RUBRIC_ITEMS"),
        ("score/4_1_1_mutual_score.py", "RUBRIC_ITEMS"),
        ("score/4_1_converge.py", "RUBRIC_ITEMS"),
        ("score/4_2_2_mutual_score.py", "RUBRIC_ITEMS"),
        ("score/4_2_2_converge.py", "RUBRIC_ITEMS"),
        ("score/2_2_converge.py", "RUBRIC_SECTIONS"),
    ],
    "question_dimensions": [
        ("score/2_score.py", "QUESTION_DIMENSIONS"),
        ("score/2_1_converge.py", "RUBRIC_QUESTIONS"),
        ("score/4_2_1_mutual_score.py", "RUBRIC_QUESTIONS"),
        ("score/4_2_1_converge.py", "RUBRIC_QUESTIONS"),
    ],
    "converge_groups": [
        ("score/2_1_converge.py", "DIM"),
        ("score/4_2_1_converge.py", "DIM"),
    ],
}


def get_question_bank_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get(QUESTION_BANK_ENV)
    return Path(env_path) if env_path else DEFAULT_QUESTION_BANK_PATH


def load_question_bank(path: str | Path | None = None) -> dict[str, Any] | None:
    bank_path = get_question_bank_path(path)
    if not bank_path.exists():
        return None
    with bank_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_hardcoded_backup(path: str | Path | None = None) -> dict[str, Any] | None:
    backup_path = Path(path) if path is not None else DEFAULT_HARDCODED_BACKUP_PATH
    if not backup_path.exists():
        return None
    with backup_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_backup_value(key: str) -> Any | None:
    backup = load_hardcoded_backup()
    if not backup:
        return None
    source_scripts = backup.get("source_scripts", {})
    for script_path, constant_name in LEGACY_BACKUP_KEY_PATHS.get(key, []):
        script_values = source_scripts.get(script_path, {})
        if constant_name in script_values:
            return script_values[constant_name]
    return None


def load_question_bank_value(
    key: str,
    default: Any,
    path: str | Path | None = None,
) -> Any:
    bank = load_question_bank(path)
    if bank:
        value = bank.get(key)
        if value not in (None, "", [], {}):
            return value
    backup_value = load_backup_value(key)
    if backup_value not in (None, "", [], {}):
        return backup_value
    return default


def save_question_bank(bank: dict[str, Any], path: str | Path | None = None) -> Path:
    bank_path = get_question_bank_path(path)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    with bank_path.open("w", encoding="utf-8") as file:
        json.dump(bank, file, ensure_ascii=False, indent=2)
    return bank_path


def question_bank_metadata() -> dict[str, str]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1",
    }


def parse_rubric_dimensions(criteria_text: str) -> list[dict[str, str]]:
    dimensions: list[dict[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in criteria_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        is_title = not raw_line.startswith((" ", "\t")) and "：" not in stripped and ":" not in stripped
        if is_title:
            if current_title is not None:
                dimensions.append(
                    {
                        "criteria": current_title,
                        "expound": "\n".join(current_lines).strip(),
                    }
                )
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(stripped)

    if current_title is not None:
        dimensions.append(
            {
                "criteria": current_title,
                "expound": "\n".join(current_lines).strip(),
            }
        )

    return dimensions


def load_question_dimensions(default: list[str], path: str | Path | None = None) -> list[str]:
    return load_question_bank_value("question_dimensions", default, path)


def load_converge_groups(default: list[list[int]], path: str | Path | None = None) -> list[list[int]]:
    return load_question_bank_value("converge_groups", default, path)


def load_rubric_dimensions(
    default: list[dict[str, str]],
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    return load_question_bank_value("rubric_dimensions", default, path)


def load_rubric_text(default: str, key: str = "rubric_text_full", path: str | Path | None = None) -> str:
    return load_question_bank_value(key, default, path)
