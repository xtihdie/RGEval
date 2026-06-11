from __future__ import annotations

from ..datasets import ESSAY_DATASET, EvaluationDataset
from ..pipeline import ESSAY_FAMILY, EvaluationFamily


def get_essay_dataset() -> EvaluationDataset:
    return ESSAY_DATASET


def get_essay_family() -> EvaluationFamily:
    return ESSAY_FAMILY
