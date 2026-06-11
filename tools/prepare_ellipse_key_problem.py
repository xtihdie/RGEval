from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_CSV = (
    PROJECT_ROOT
    / "data"
    / "essay"
    / "origin"
    / "ELLIPSE-Corpus-main"
    / "ELLIPSE_Final_github_train.csv"
)
DEFAULT_ESSAY_ROOT = PROJECT_ROOT / "data" / "essay"
DEFAULT_CRITERIA_PATH = PROJECT_ROOT / "text" / "essay_criteria_ellipse.txt"

TRAIT_RUBRIC_TREE = [
    {
        "trait": "Cohesion",
        "description": "Evaluate how clearly ideas connect across the whole essay.",
        "leaves": [
            {
                "label": "Logical progression",
                "description": "Evaluate whether ideas and claims progress in a sensible order from beginning to end.",
            },
            {
                "label": "Cross-sentence and cross-paragraph linking",
                "description": "Evaluate whether transitions and links between sentences or paragraphs make relationships easy to follow.",
            },
            {
                "label": "Reference clarity",
                "description": "Evaluate whether pronouns, repeated references, and topic mentions stay clear rather than confusing.",
            },
        ],
    },
    {
        "trait": "Syntax",
        "description": "Evaluate control of sentence structure and syntactic clarity.",
        "leaves": [
            {
                "label": "Sentence completeness and well-formedness",
                "description": "Evaluate whether sentences are grammatically complete and structurally well formed.",
            },
            {
                "label": "Sentence variety and complexity",
                "description": "Evaluate whether the essay uses an effective range of simple, compound, and complex sentence structures.",
            },
            {
                "label": "Word order and clause-level clarity",
                "description": "Evaluate whether word order and clause arrangement support clear meaning without syntactic confusion.",
            },
        ],
    },
    {
        "trait": "Vocabulary",
        "description": "Evaluate lexical choice, range, and precision.",
        "leaves": [
            {
                "label": "Lexical precision",
                "description": "Evaluate whether words are chosen accurately to express the intended meaning.",
            },
            {
                "label": "Lexical range",
                "description": "Evaluate whether the essay shows enough vocabulary range instead of relying on very repetitive wording.",
            },
            {
                "label": "Register and task appropriateness",
                "description": "Evaluate whether vocabulary fits the task, audience, and argumentative or explanatory purpose.",
            },
        ],
    },
    {
        "trait": "Phraseology",
        "description": "Evaluate the naturalness and fluency of phrase-level expression.",
        "leaves": [
            {
                "label": "Collocation naturalness",
                "description": "Evaluate whether common word combinations sound natural rather than awkward.",
            },
            {
                "label": "Phrasal fluency",
                "description": "Evaluate whether multi-word expressions read smoothly and support fluent written expression.",
            },
            {
                "label": "Awkward or literal phrasing avoidance",
                "description": "Evaluate whether the essay avoids translated, literal, or unnatural phrasing patterns.",
            },
        ],
    },
    {
        "trait": "Grammar",
        "description": "Evaluate grammatical accuracy at the sentence and clause level.",
        "leaves": [
            {
                "label": "Morphology and agreement",
                "description": "Evaluate tense, number, subject-verb agreement, and related form-level grammar accuracy.",
            },
            {
                "label": "Clause-level grammar accuracy",
                "description": "Evaluate whether clauses are built accurately without major grammatical breakdowns.",
            },
            {
                "label": "Error frequency and impact",
                "description": "Evaluate how often grammatical errors occur and how strongly they interfere with meaning.",
            },
        ],
    },
    {
        "trait": "Conventions",
        "description": "Evaluate writing mechanics and surface correctness.",
        "leaves": [
            {
                "label": "Spelling accuracy",
                "description": "Evaluate whether spelling errors are infrequent and do not distract the reader.",
            },
            {
                "label": "Punctuation and capitalization",
                "description": "Evaluate whether punctuation and capitalization are used correctly and consistently.",
            },
            {
                "label": "Mechanics consistency",
                "description": "Evaluate whether surface conventions remain consistent enough to preserve readability.",
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare ELLIPSE for rubric-tree-based essay key-problem generation")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--essay-root", default=str(DEFAULT_ESSAY_ROOT))
    parser.add_argument("--criteria-path", default=str(DEFAULT_CRITERIA_PATH))
    return parser.parse_args()


def build_rubric_tree() -> dict:
    tree = {
        "dataset_name": "ELLIPSE",
        "source_name": "ELLIPSE-Corpus",
        "mode": "rubric_tree_draft",
        "score_scale": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "overall_score_column": "Overall",
        "trait_score_columns": [item["trait"] for item in TRAIT_RUBRIC_TREE],
        "top_level_traits": TRAIT_RUBRIC_TREE,
    }
    return tree


def render_criteria_text() -> str:
    lines = [
        "ELLIPSE Essay Analytic Scoring",
        "\tUse the official analytic traits as top-level rubric dimensions, then refine each trait into leaf rubrics before generating essay key problems.",
    ]
    for trait in TRAIT_RUBRIC_TREE:
        lines.append(trait["trait"])
        lines.append(f"\t{trait['description']}")
        for leaf in trait["leaves"]:
            lines.append(f"\t{leaf['label']}: {leaf['description']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    source_csv = Path(args.source_csv)
    essay_root = Path(args.essay_root)
    criteria_path = Path(args.criteria_path)
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing ELLIPSE source CSV: {source_csv}")

    origin_dir = essay_root / "origin"
    results_dir = essay_root / "results"
    origin_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)

    normalized = df[
        [
            "text_id_kaggle",
            "full_text",
            "prompt",
            "task",
            "grade",
            "gender",
            "set",
            "Overall",
            "Cohesion",
            "Syntax",
            "Vocabulary",
            "Phraseology",
            "Grammar",
            "Conventions",
        ]
    ].copy()
    normalized = normalized.rename(
        columns={
            "text_id_kaggle": "essay_id",
            "set": "split",
            "Overall": "overall_score",
            "Cohesion": "cohesion",
            "Syntax": "syntax",
            "Vocabulary": "vocabulary",
            "Phraseology": "phraseology",
            "Grammar": "grammar",
            "Conventions": "conventions",
        }
    )

    normalized_path = origin_dir / "ellipse_train_normalized.csv"
    normalized.to_csv(normalized_path, index=False, encoding="utf-8-sig")

    score_df = normalized[
        [
            "essay_id",
            "overall_score",
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
            "split",
        ]
    ].copy()
    score_path = essay_root / "score.csv"
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")

    rubric_tree = build_rubric_tree()
    rubric_tree_path = essay_root / "rubric_tree_ellipse.json"
    rubric_tree_path.write_text(
        json.dumps(rubric_tree, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    criteria_path.parent.mkdir(parents=True, exist_ok=True)
    criteria_path.write_text(render_criteria_text(), encoding="utf-8")

    print(f"Saved normalized essays to: {normalized_path}")
    print(f"Saved score table to: {score_path}")
    print(f"Saved rubric tree draft to: {rubric_tree_path}")
    print(f"Saved essay criteria draft to: {criteria_path}")


if __name__ == "__main__":
    main()
