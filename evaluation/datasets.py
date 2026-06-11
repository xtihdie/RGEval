from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .paths import DATA_DIR


@dataclass(frozen=True)
class EvaluationDataset:
    key: str
    display_name: str
    source_type: str
    root_dir: Path
    raw_dir: Path
    results_dir: Path
    gold_path: Path | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


CLASSROOM_DIALOGUE_DATASET = EvaluationDataset(
    key="classroom_dialogue",
    display_name="Classroom Dialogue Evaluation",
    source_type="dialogue_csv",
    root_dir=DATA_DIR / "300class",
    raw_dir=DATA_DIR / "300class" / "origin",
    results_dir=DATA_DIR / "300class" / "results",
    gold_path=DATA_DIR / "300class" / "score.csv",
    notes=(
        "Legacy score scripts already target this dataset.",
        "This remains the default dataset for the current project.",
    ),
)


ESSAY_DATASET = EvaluationDataset(
    key="essay",
    display_name="Essay Evaluation",
    source_type="essay_table",
    root_dir=DATA_DIR / "essay",
    raw_dir=DATA_DIR / "essay" / "origin",
    results_dir=DATA_DIR / "essay" / "results",
    gold_path=DATA_DIR / "essay" / "score.csv",
    notes=(
        "Reserved for the next evaluation task.",
        "The pipeline is intended to mirror the classroom dialogue workflow.",
        "Only dataset shape and prompts should differ in most cases.",
    ),
)


WIKI_QUALITY_DATASET = EvaluationDataset(
    key="wiki_quality",
    display_name="Wikipedia Quality Assessment",
    source_type="wiki_article_table",
    root_dir=DATA_DIR / "wiki_quality",
    raw_dir=DATA_DIR / "wiki_quality" / "origin",
    results_dir=DATA_DIR / "wiki_quality" / "results",
    gold_path=DATA_DIR / "wiki_quality" / "metadata.csv",
    notes=(
        "Uses the 2017 English Wikipedia quality dataset.",
        "Current comparison only includes direct scoring and one-level decomposition.",
    ),
)


DATASETS: dict[str, EvaluationDataset] = {
    CLASSROOM_DIALOGUE_DATASET.key: CLASSROOM_DIALOGUE_DATASET,
    ESSAY_DATASET.key: ESSAY_DATASET,
    WIKI_QUALITY_DATASET.key: WIKI_QUALITY_DATASET,
}


def get_dataset(key: str) -> EvaluationDataset:
    try:
        return DATASETS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {key!r}. Supported datasets: {supported}") from exc
