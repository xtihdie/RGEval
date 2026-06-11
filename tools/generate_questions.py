from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.question_bank import (  # noqa: E402
    parse_rubric_dimensions,
    question_bank_metadata,
    save_question_bank,
)
from evaluation.prompt_sets import CLASSROOM_PROMPTS, ESSAY_PROMPTS  # noqa: E402
from llm_pool.runner import ThreadRunner  # noqa: E402


DEFAULT_CRITERIA_PATH = PROJECT_ROOT / "text" / "criteria.txt"
DEFAULT_LEGACY_OUTPUT_PATH = PROJECT_ROOT / "text" / "question.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate and persist question bank for scoring pipelines")
    parser.add_argument("--agent", default="zhipu", help="Provider alias from config.toml")
    parser.add_argument("--model", default=None, help="Model id or semantic alias. Defaults to the provider's configured default model.")
    parser.add_argument("--synonym-agent", default=None, help="Optional provider override for merge step")
    parser.add_argument("--synonym-model", default=None, help="Optional model override for merge step")
    parser.add_argument("--depth", type=int, default=2, help="Tree expansion depth")
    parser.add_argument("--criteria-path", default=str(DEFAULT_CRITERIA_PATH))
    parser.add_argument("--legacy-output-path", default=str(DEFAULT_LEGACY_OUTPUT_PATH))
    parser.add_argument(
        "--prompt-family",
        choices=("auto", "classroom", "essay"),
        default="auto",
        help="Select which prompt family to use for rubric decomposition and question generation.",
    )
    return parser.parse_args()


def resolve_prompt_family(criteria_path: Path, requested: str) -> dict:
    if requested == "classroom":
        return CLASSROOM_PROMPTS
    if requested == "essay":
        return ESSAY_PROMPTS

    lowered = str(criteria_path).lower()
    if "essay" in lowered or "ellipse" in lowered:
        return ESSAY_PROMPTS
    return CLASSROOM_PROMPTS


def query_once(runner: ThreadRunner, system_prompt: str, user_prompt: str) -> str:
    messages = [[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]]
    return runner.run(messages)[0]


def parse_star_numbered_items(raw: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in raw.splitlines():
        match = re.match(r"^\*{0,3}\s*(\d+)\.\s*(.+?)\s*$", line.strip())
        if match:
            items.append((match.group(1), match.group(2)))
    if not items:
        raise ValueError(f"Unable to parse numbered items from response:\n{raw}")
    return items


def generate_tree_dict(criteria: str, depth: int, runner: ThreadRunner, prompts: dict) -> dict[str, str]:
    response_history = {"1": criteria}
    index = 0

    while index < len(response_history):
        current_key = list(response_history.keys())[index]
        current_value = response_history[current_key]
        if len(current_key.split(".")) >= depth:
            index += 1
            continue

        response = query_once(
            runner,
            prompts["evaluation_criteria_division"]["system"],
            prompts["evaluation_criteria_division"]["user"].format(criteria=current_value),
        )
        parsed_items = parse_star_numbered_items(response)

        if len(parsed_items) == 1:
            index += 1
            continue

        for item_index, item_text in parsed_items:
            response_history[f"{current_key}.{item_index.strip()}"] = item_text.strip()
        index += 1

    return response_history


def parse_tag_block(text: str, tag_name: str) -> str:
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"Missing <{tag_name}> block in response:\n{text}")
    return match.group(1).strip()


def parse_question_list_block(block: str) -> list[str]:
    questions: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        _, question_text = line.split("-", 1)
        question_text = question_text.strip()
        if not question_text.endswith("？"):
            question_text = question_text.rstrip("?.") + "？"
        questions.append(question_text)
    return questions


def parse_change_table_block(block: str, question_count: int) -> list[list[int]]:
    change_table = [[] for _ in range(question_count)]
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        new_index_text, old_indices_text = line.split(":", 1)
        new_index = int(new_index_text.strip())
        old_indices = [int(value) for value in re.findall(r"\d+", old_indices_text)]
        for old_index in old_indices:
            if 1 <= old_index <= question_count and new_index != 0:
                change_table[old_index - 1].append(new_index)
    return change_table


def dict_to_question_bank(
    criteria: str,
    depth: int,
    response_history: dict[str, str],
    question_runner: ThreadRunner,
    synonym_runner: ThreadRunner,
    prompts: dict,
) -> dict:
    dict_keys = list(response_history.keys())[1:]
    node_question_table: dict[str, list[int]] = {}
    old_question_list: list[str] = []

    for key in dict_keys:
        if len(key.split(".")) != depth:
            continue
        response = query_once(
            question_runner,
            prompts["evaluation_criteria_question"]["system"],
            prompts["evaluation_criteria_question"]["user"].format(criteria=response_history[key]),
        )
        question_items = [text for _, text in parse_star_numbered_items(response)]
        node_question_table[key] = [len(old_question_list) + index + 1 for index in range(len(question_items))]
        old_question_list.extend(question_items)

    numbered_old_questions = "\n".join(
        f"{index + 1}-{question}" for index, question in enumerate(old_question_list)
    )
    merge_response = query_once(
        synonym_runner,
        prompts["question_synonym"]["system"],
        prompts["question_synonym"]["user"].format(criteria=criteria, question_list=numbered_old_questions),
    )

    new_question_list = parse_question_list_block(parse_tag_block(merge_response, "question_list"))
    change_table = parse_change_table_block(
        parse_tag_block(merge_response, "change_table"),
        question_count=len(old_question_list),
    )

    normalized_node_question_table: dict[str, list[int]] = {}
    for key, original_indices in node_question_table.items():
        merged_indices = [
            merged
            for original_index in original_indices
            for merged in change_table[original_index - 1]
        ]
        normalized_node_question_table[key] = sorted(set(merged_indices))

    rubric_dimensions = parse_rubric_dimensions(criteria)

    bank = {
        **question_bank_metadata(),
        "criteria_text": criteria,
        "depth": depth,
        "rubric_dimensions": rubric_dimensions,
        "question_dimensions": new_question_list,
        "leaf_question_map": normalized_node_question_table,
        "converge_groups": [],
        "tree": response_history,
    }
    return bank


def render_legacy_snapshot(bank: dict) -> str:
    lines = []
    for index, question in enumerate(bank.get("question_dimensions", []), 1):
        lines.append(f"{index: <2} - {question}")
    lines.append("")
    lines.append(str(bank.get("leaf_question_map", {})))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    criteria_path = Path(args.criteria_path)
    legacy_output_path = Path(args.legacy_output_path)
    criteria = criteria_path.read_text(encoding="utf-8")
    prompt_bundle = resolve_prompt_family(criteria_path, args.prompt_family)

    divide_runner = ThreadRunner(args.agent, args.model, max_workers=1)
    question_runner = ThreadRunner(args.agent, args.model, max_workers=1)
    synonym_runner = ThreadRunner(
        args.synonym_agent or args.agent,
        args.synonym_model or args.model,
        max_workers=1,
    )

    tree = generate_tree_dict(criteria=criteria, depth=args.depth, runner=divide_runner, prompts=prompt_bundle)
    bank = dict_to_question_bank(
        criteria=criteria,
        depth=args.depth,
        response_history=tree,
        question_runner=question_runner,
        synonym_runner=synonym_runner,
        prompts=prompt_bundle,
    )

    bank_path = save_question_bank(bank)
    legacy_snapshot = render_legacy_snapshot(bank)
    legacy_output_path.write_text(legacy_snapshot, encoding="utf-8")

    print(legacy_snapshot)
    print(f"\nSaved question bank to: {bank_path}")
    print(f"Saved legacy snapshot to: {legacy_output_path}")


if __name__ == "__main__":
    main()
