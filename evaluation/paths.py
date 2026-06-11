from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEGACY_SCORE_DIR = PROJECT_ROOT / "score"
TOOLS_DIR = PROJECT_ROOT / "tools"
TEXT_DIR = PROJECT_ROOT / "text"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
