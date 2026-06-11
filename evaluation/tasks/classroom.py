from __future__ import annotations

from ..datasets import CLASSROOM_DIALOGUE_DATASET, EvaluationDataset
from ..pipeline import CLASSROOM_DIALOGUE_FAMILY, EvaluationFamily


def get_classroom_dataset() -> EvaluationDataset:
    return CLASSROOM_DIALOGUE_DATASET


def get_classroom_family() -> EvaluationFamily:
    return CLASSROOM_DIALOGUE_FAMILY
