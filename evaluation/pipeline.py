from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationStage:
    key: str
    display_name: str
    purpose: str
    legacy_scripts: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationFamily:
    key: str
    display_name: str
    dataset_key: str
    stages: tuple[EvaluationStage, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


CLASSROOM_DIALOGUE_FAMILY = EvaluationFamily(
    key="classroom_dialogue",
    display_name="Classroom Dialogue Pipeline",
    dataset_key="classroom_dialogue",
    stages=(
        EvaluationStage(
            key="lesson_score",
            display_name="Lesson Score",
            purpose="Produce initial lesson-level or dimension-level judgments from source dialogue.",
            legacy_scripts=("score/0_score.py", "score/1_score.py", "score/2_score.py"),
        ),
        EvaluationStage(
            key="converge",
            display_name="Converge",
            purpose="Aggregate prior judgments into a more stable conclusion.",
            legacy_scripts=(
                "score/1_converge.py",
                "score/2_1_converge.py",
                "score/2_2_converge.py",
                "score/4_1_converge.py",
                "score/4_2_1_converge.py",
                "score/4_2_2_converge.py",
            ),
        ),
        EvaluationStage(
            key="mutual_review",
            display_name="Mutual Review",
            purpose="Use cross-model review to refine judgments and reduce single-model bias.",
            legacy_scripts=(
                "score/4_0_mutual_score.py",
                "score/4_1_1_mutual_score.py",
                "score/4_1_2_mutual_score.py",
                "score/4_2_1_mutual_score.py",
                "score/4_2_2_mutual_score.py",
                "score/4_2_3_mutual_score.py",
            ),
        ),
        EvaluationStage(
            key="cleanup",
            display_name="Cleanup",
            purpose="Detect invalid rows and prepare reruns for missing work.",
            legacy_scripts=("score/3_cleanup.py",),
        ),
    ),
    notes=(
        "This is the current production pipeline.",
        "Legacy script names are preserved for compatibility.",
    ),
)


ESSAY_FAMILY = EvaluationFamily(
    key="essay",
    display_name="Essay Evaluation Pipeline",
    dataset_key="essay",
    stages=(
        EvaluationStage(
            key="essay_score",
            display_name="Essay Score",
            purpose="Run first-pass essay evaluation on essay records.",
        ),
        EvaluationStage(
            key="essay_converge",
            display_name="Essay Converge",
            purpose="Aggregate earlier essay judgments into final outputs.",
        ),
        EvaluationStage(
            key="essay_mutual_review",
            display_name="Essay Mutual Review",
            purpose="Optionally add cross-model review if essay outputs need the same robustness layer.",
        ),
    ),
    notes=(
        "Reserved for the next phase of the project.",
        "The family intentionally mirrors classroom stages so shared code can be reused.",
    ),
)


FAMILIES: dict[str, EvaluationFamily] = {
    CLASSROOM_DIALOGUE_FAMILY.key: CLASSROOM_DIALOGUE_FAMILY,
    ESSAY_FAMILY.key: ESSAY_FAMILY,
}


def get_family(key: str) -> EvaluationFamily:
    try:
        return FAMILIES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(FAMILIES))
        raise KeyError(f"Unknown family {key!r}. Supported families: {supported}") from exc
