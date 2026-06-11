from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


QUESTION_BANK_ENV = "EVALUATION_QUESTION_BANK_PATH"


def add_path_override_argument(
    parser: ArgumentParser,
    flag: str,
    dest: str,
    help_text: str,
) -> None:
    parser.add_argument(flag, dest=dest, type=str, default=None, help=help_text)


def add_question_bank_argument(parser: ArgumentParser) -> None:
    add_path_override_argument(
        parser,
        "--question-bank-path",
        "question_bank_path",
        "Override the question bank JSON path for this run.",
    )


def set_question_bank_override(args: Namespace) -> None:
    question_bank_path = getattr(args, "question_bank_path", None)
    if question_bank_path:
        os.environ[QUESTION_BANK_ENV] = question_bank_path


def override_config_value(cfg: Any, args: Namespace, attr_name: str, *, as_path: bool = False) -> None:
    value = getattr(args, attr_name, None)
    if value is None:
        return
    setattr(cfg, attr_name, Path(value) if as_path else value)
