from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESSAY_DIR = PROJECT_ROOT / "data" / "essay"
QUESTION_BANK_PATH = ESSAY_DIR / "question_bank.json"
RAW_BACKUP_PATH = ESSAY_DIR / "question_bank.generated_raw.json"
RUBRIC_TREE_PATH = ESSAY_DIR / "rubric_tree_ellipse.json"


TRAIT_LAYOUT = [
    ("cohesion", "Cohesion", range(1, 5)),
    ("syntax", "Syntax", range(5, 8)),
    ("vocabulary", "Vocabulary", range(8, 11)),
    ("phraseology", "Phraseology", range(11, 14)),
    ("grammar", "Grammar", range(14, 17)),
    ("conventions", "Conventions", range(17, 20)),
]

FIXED_ELLIPSE_QUESTION_TRAITS = [
    "cohesion",
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]

FIXED_ELLIPSE_GROUPS = [
    [1, 2],
    [3],
    [4],
    [5],
    [6],
    [7],
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_rubric_dimensions(rubric_tree: dict) -> list[dict[str, str]]:
    dimensions: list[dict[str, str]] = []
    for trait in rubric_tree["top_level_traits"]:
        leaf_lines = [f"{leaf['label']}: {leaf['description']}" for leaf in trait["leaves"]]
        dimensions.append(
            {
                "criteria": trait["trait"],
                "expound": "\n".join([trait["description"], *leaf_lines]),
            }
        )
    return dimensions


def clean_question(question: str) -> str:
    cleaned = (question or "").strip()
    cleaned = cleaned.replace("é”›?", "?").replace("？", "?")
    cleaned = re.sub(r"[^\x00-\x7F?]+$", "", cleaned).strip()
    if cleaned and not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".") + "?"
    return cleaned


def build_trait_groups(filtered_leaf_map: dict[str, list[int]]) -> tuple[list[str], list[list[int]]]:
    max_question_id = max((max(ids) for ids in filtered_leaf_map.values() if ids), default=0)
    if max_question_id == 7:
        return FIXED_ELLIPSE_QUESTION_TRAITS, FIXED_ELLIPSE_GROUPS

    question_traits: list[str] = []
    converge_groups: list[list[int]] = []
    for trait_key, _, leaf_range in TRAIT_LAYOUT:
        collected: list[int] = []
        for leaf_index in leaf_range:
            collected.extend(filtered_leaf_map.get(f"1.{leaf_index}", []))
        deduped = sorted(set(collected))
        if deduped:
            converge_groups.append(deduped)
            question_traits.extend([trait_key] * 0)

    question_count = max_question_id
    question_traits = [""] * question_count
    for trait_key, _, leaf_range in TRAIT_LAYOUT:
        collected: list[int] = []
        for leaf_index in leaf_range:
            collected.extend(filtered_leaf_map.get(f"1.{leaf_index}", []))
        for qid in sorted(set(collected)):
            question_traits[qid - 1] = trait_key
    return question_traits, converge_groups


def main() -> None:
    bank = load_json(QUESTION_BANK_PATH)
    rubric_tree = load_json(RUBRIC_TREE_PATH)

    if not RAW_BACKUP_PATH.exists():
        dump_json(RAW_BACKUP_PATH, bank)

    kept_questions: list[str] = []
    id_remap: dict[int, int] = {}
    for original_id, question in enumerate(bank.get("question_dimensions", []), start=1):
        question = clean_question(question)
        new_id = len(kept_questions) + 1
        kept_questions.append(question)
        id_remap[original_id] = new_id

    filtered_leaf_map: dict[str, list[int]] = {}
    for leaf_key, question_ids in bank.get("leaf_question_map", {}).items():
        new_ids = sorted({id_remap[qid] for qid in question_ids if qid in id_remap})
        if new_ids:
            filtered_leaf_map[leaf_key] = new_ids

    filtered_tree = dict(bank.get("tree", {}))
    question_traits, converge_groups = build_trait_groups(filtered_leaf_map)

    curated = dict(bank)
    curated["question_dimensions"] = kept_questions
    curated["leaf_question_map"] = filtered_leaf_map
    curated["rubric_dimensions"] = build_rubric_dimensions(rubric_tree)
    curated["question_traits"] = question_traits
    curated["trait_question_groups"] = converge_groups
    curated["converge_groups"] = converge_groups
    curated["dataset_name"] = rubric_tree.get("dataset_name", "ELLIPSE")
    curated["source_name"] = rubric_tree.get("source_name", "ELLIPSE-Corpus")
    curated["score_scale"] = rubric_tree.get("score_scale", [])
    curated["overall_score_column"] = rubric_tree.get("overall_score_column", "Overall")
    curated["trait_score_columns"] = rubric_tree.get("trait_score_columns", [])
    curated["tree"] = filtered_tree
    curated["curation_notes"] = [
        "Cleaned malformed punctuation from generated English key questions.",
        "Aligned question-to-trait and converge groups with the six official ELLIPSE analytic traits.",
        "Replaced rubric dimensions with the six official top-level traits from the curated rubric tree draft.",
    ]

    dump_json(QUESTION_BANK_PATH, curated)
    print(f"Curated question bank written to: {QUESTION_BANK_PATH}")
    print(f"Raw generated backup kept at: {RAW_BACKUP_PATH}")


if __name__ == "__main__":
    main()
