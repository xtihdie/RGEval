from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from evaluation.datasets import CLASSROOM_DIALOGUE_DATASET, get_dataset
from evaluation.paths import PROJECT_ROOT


@dataclass(frozen=True)
class RuntimePaths:
    dataset_root: Path
    raw_dir: Path
    results_dir: Path
    gold_path: Path
    question_bank_path: Path | None


SCORE_DIR = PROJECT_ROOT / "score"

CORE_STAGE_TO_SCRIPT = {
    "0": "0_score.py",
    "1": "1_score.py",
    "1_converge": "1_converge.py",
    "2": "2_score.py",
    "2_1_converge": "2_1_converge.py",
    "2_2_converge": "2_2_converge.py",
}

ESSAY_STAGE_TO_SCRIPT = {
    "essay_0": "essay_0_score.py",
    "essay_1": "essay_1_score.py",
    "essay_1_converge": "essay_1_converge.py",
    "essay_2": "essay_2_score.py",
    "essay_2_converge": "essay_2_converge.py",
}

WIKI_STAGE_TO_SCRIPT = {
    "wiki_0": "wiki_0_score.py",
    "wiki_2": "wiki_2_score.py",
    "wiki_2_converge": "wiki_2_converge.py",
}

PAIR_STAGE_TO_SCRIPT = {
    "4_0": "4_0_mutual_score.py",
    "4_1_1": "4_1_1_mutual_score.py",
    "4_1_2": "4_1_2_mutual_score.py",
    "4_2_1": "4_2_1_mutual_score.py",
    "4_2_2": "4_2_2_mutual_score.py",
    "4_2_3": "4_2_3_mutual_score.py",
}

SINGLE_STAGE_TO_SCRIPT = {
    "4_1_converge": "4_1_converge.py",
    "4_2_1_converge": "4_2_1_converge.py",
    "4_2_2_converge": "4_2_2_converge.py",
}

STAGE_CHOICES = (
    "0",
    "1",
    "1_converge",
    "2",
    "2_1_converge",
    "2_2_converge",
    "essay_0",
    "essay_1",
    "essay_1_converge",
    "essay_2",
    "essay_2_converge",
    "wiki_0",
    "wiki_2",
    "wiki_2_converge",
    "core",
    "4_0",
    "4_1_1",
    "4_1_converge",
    "4_1_2",
    "4_1",
    "4_2_1",
    "4_2_1_converge",
    "4_2_2",
    "4_2_2_converge",
    "4_2_3",
    "4_2",
    "full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified evaluation entrypoint")
    parser.add_argument("--dataset", type=str, default=CLASSROOM_DIALOGUE_DATASET.key)
    parser.add_argument("--stage", type=str, default="core", choices=STAGE_CHOICES)

    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--gold-path", type=str, default=None)
    parser.add_argument("--question-bank-path", type=str, default=None)

    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--agent-a", type=str, default=None)
    parser.add_argument("--model-a", type=str, default=None)
    parser.add_argument("--agent-b", type=str, default=None)
    parser.add_argument("--model-b", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-essays", type=int, default=None)
    parser.add_argument("--essay-id", action="append", dest="essay_ids", default=None)

    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    dataset = get_dataset(args.dataset)
    dataset_root = Path(args.dataset_root) if args.dataset_root else dataset.root_dir
    raw_dir = Path(args.raw_dir) if args.raw_dir else dataset_root / "origin"
    results_dir = Path(args.results_dir) if args.results_dir else dataset_root / "results"
    gold_path = Path(args.gold_path) if args.gold_path else (dataset.gold_path or dataset_root / "score.csv")
    question_bank_path = (
        Path(args.question_bank_path)
        if args.question_bank_path
        else dataset_root / "question_bank.json"
    )
    if question_bank_path.exists():
        qb = question_bank_path
    else:
        qb = None
    return RuntimePaths(
        dataset_root=dataset_root,
        raw_dir=raw_dir,
        results_dir=results_dir,
        gold_path=gold_path,
        question_bank_path=qb,
    )


def resolve_single_agent(args: argparse.Namespace) -> tuple[str, str]:
    agent = args.agent or args.agent_a
    model = args.model or args.model_a
    if not agent or not model:
        raise ValueError("This stage requires --agent/--model, or it can fall back to --agent-a/--model-a.")
    return agent, model


def resolve_pair(args: argparse.Namespace) -> tuple[tuple[str, str], tuple[str, str]]:
    if not all([args.agent_a, args.model_a, args.agent_b, args.model_b]):
        raise ValueError("This stage requires --agent-a, --model-a, --agent-b, and --model-b.")
    return (args.agent_a, args.model_a), (args.agent_b, args.model_b)


def run_command(command: list[str], *, dry_run: bool) -> None:
    pretty = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(pretty)
    if not dry_run:
        env = dict(__import__("os").environ)
        existing = env.get("PYTHONPATH", "")
        root_text = str(PROJECT_ROOT)
        env["PYTHONPATH"] = root_text if not existing else root_text + __import__("os").pathsep + existing
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def extend_if_present(parts: list[str], flag: str, value: str | Path | None) -> None:
    if value is None:
        return
    parts.extend([flag, str(value)])


def build_core_command(
    script_name: str,
    agent: str,
    model: str,
    paths: RuntimePaths,
    num_threads: int | None,
) -> list[str]:
    command = [sys.executable, str(SCORE_DIR / script_name), "--agent", agent, "--model", model]
    extend_if_present(command, "--class-file-path", paths.raw_dir)
    extend_if_present(command, "--score-file-path", paths.gold_path)
    extend_if_present(command, "--result-dir", paths.results_dir)
    extend_if_present(command, "--question-bank-path", paths.question_bank_path)
    extend_if_present(command, "--num-threads", num_threads)
    return command


def build_essay_command(
    script_name: str,
    agent: str,
    model: str,
    paths: RuntimePaths,
    args: argparse.Namespace,
    num_threads: int | None,
) -> list[str]:
    command = [sys.executable, str(SCORE_DIR / script_name), "--agent", agent, "--model", model]
    extend_if_present(command, "--essay-file-path", paths.raw_dir / "ellipse_train_normalized.csv")
    extend_if_present(command, "--score-file-path", paths.gold_path)
    extend_if_present(command, "--result-dir", paths.results_dir)
    extend_if_present(command, "--question-bank-path", paths.question_bank_path)
    extend_if_present(command, "--num-threads", num_threads)
    extend_if_present(command, "--split", args.split)
    extend_if_present(command, "--max-essays", args.max_essays)
    if args.essay_ids:
        for essay_id in args.essay_ids:
            extend_if_present(command, "--essay-id", essay_id)
    return command


def build_wiki_command(
    script_name: str,
    agent: str,
    model: str,
    paths: RuntimePaths,
    args: argparse.Namespace,
    num_threads: int | None,
) -> list[str]:
    command = [sys.executable, str(SCORE_DIR / script_name), "--agent", agent, "--model", model]
    extend_if_present(command, "--metadata-file-path", paths.gold_path)
    extend_if_present(command, "--result-dir", paths.results_dir)
    extend_if_present(command, "--question-bank-path", paths.question_bank_path)
    extend_if_present(command, "--num-threads", num_threads)
    extend_if_present(command, "--split", args.split)
    extend_if_present(command, "--max-articles", args.max_essays)
    if args.essay_ids:
        for article_id in args.essay_ids:
            extend_if_present(command, "--article-id", article_id)
    return command


def build_pair_command(
    script_name: str,
    pair: tuple[tuple[str, str], tuple[str, str]],
    paths: RuntimePaths,
    num_threads: int | None,
) -> list[str]:
    (agent_a, model_a), (agent_b, model_b) = pair
    command = [
        sys.executable,
        str(SCORE_DIR / script_name),
        "--agent-a",
        agent_a,
        "--model-a",
        model_a,
        "--agent-b",
        agent_b,
        "--model-b",
        model_b,
    ]
    extend_if_present(command, "--num-threads", num_threads)

    stage_dir = None
    if script_name == "4_0_mutual_score.py":
        extend_if_present(command, "--class-file-path", paths.raw_dir)
        stage_dir = paths.results_dir / "4" / "0"
        extend_if_present(command, "--prev-score-dir", paths.results_dir / "0")
    elif script_name in {"4_1_1_mutual_score.py", "4_1_2_mutual_score.py"}:
        extend_if_present(command, "--class-file-path", paths.raw_dir)
        stage_dir = paths.results_dir / "4" / "1"
        if script_name == "4_1_1_mutual_score.py":
            extend_if_present(command, "--prev-dim-dir", paths.results_dir / "1")
    elif script_name in {"4_2_1_mutual_score.py", "4_2_2_mutual_score.py", "4_2_3_mutual_score.py"}:
        extend_if_present(command, "--class-file-path", paths.raw_dir)
        stage_dir = paths.results_dir / "4" / "2"
        if script_name == "4_2_1_mutual_score.py":
            extend_if_present(command, "--prev-dim-dir", paths.results_dir / "2")

    extend_if_present(command, "--result-dir", stage_dir)
    extend_if_present(command, "--question-bank-path", paths.question_bank_path)
    return command


def build_single_stage_command(
    script_name: str,
    agent: str,
    model: str,
    paths: RuntimePaths,
    num_threads: int | None,
) -> list[str]:
    command = [sys.executable, str(SCORE_DIR / script_name), "--agent", agent, "--model", model]
    extend_if_present(command, "--num-threads", num_threads)

    if script_name == "4_1_converge.py":
        extend_if_present(command, "--result-dir", paths.results_dir / "4" / "1")
        extend_if_present(command, "--step1-dir", paths.results_dir / "1")
    elif script_name in {"4_2_1_converge.py", "4_2_2_converge.py"}:
        extend_if_present(command, "--result-dir", paths.results_dir / "4" / "2")
        extend_if_present(command, "--step2-dir", paths.results_dir / "2")

    extend_if_present(command, "--question-bank-path", paths.question_bank_path)
    return command


def run_commands(commands: Iterable[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        run_command(command, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    paths = resolve_runtime_paths(args)

    stage = args.stage
    dry_run = args.dry_run
    num_threads = args.num_threads

    if stage in CORE_STAGE_TO_SCRIPT:
        agent, model = resolve_single_agent(args)
        command = build_core_command(CORE_STAGE_TO_SCRIPT[stage], agent, model, paths, num_threads)
        run_command(command, dry_run=dry_run)
        return

    if stage in ESSAY_STAGE_TO_SCRIPT:
        agent, model = resolve_single_agent(args)
        command = build_essay_command(ESSAY_STAGE_TO_SCRIPT[stage], agent, model, paths, args, num_threads)
        run_command(command, dry_run=dry_run)
        return

    if stage in WIKI_STAGE_TO_SCRIPT:
        agent, model = resolve_single_agent(args)
        command = build_wiki_command(WIKI_STAGE_TO_SCRIPT[stage], agent, model, paths, args, num_threads)
        run_command(command, dry_run=dry_run)
        return

    if stage in PAIR_STAGE_TO_SCRIPT:
        pair = resolve_pair(args)
        command = build_pair_command(PAIR_STAGE_TO_SCRIPT[stage], pair, paths, num_threads)
        run_command(command, dry_run=dry_run)
        return

    if stage in SINGLE_STAGE_TO_SCRIPT:
        agent, model = resolve_single_agent(args)
        command = build_single_stage_command(SINGLE_STAGE_TO_SCRIPT[stage], agent, model, paths, num_threads)
        run_command(command, dry_run=dry_run)
        return

    if stage == "core":
        agent, model = resolve_single_agent(args)
        commands = [
            build_core_command(CORE_STAGE_TO_SCRIPT[name], agent, model, paths, num_threads)
            for name in ("0", "1", "1_converge", "2", "2_1_converge", "2_2_converge")
        ]
        run_commands(commands, dry_run=dry_run)
        return

    if stage == "4_1":
        pair = resolve_pair(args)
        commands = [
            build_pair_command(PAIR_STAGE_TO_SCRIPT["4_1_1"], pair, paths, num_threads),
            build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_1_converge"], pair[0][0], pair[0][1], paths, num_threads),
            build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_1_converge"], pair[1][0], pair[1][1], paths, num_threads),
            build_pair_command(PAIR_STAGE_TO_SCRIPT["4_1_2"], pair, paths, num_threads),
        ]
        run_commands(commands, dry_run=dry_run)
        return

    if stage == "4_2":
        pair = resolve_pair(args)
        commands = [
            build_pair_command(PAIR_STAGE_TO_SCRIPT["4_2_1"], pair, paths, num_threads),
            build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_1_converge"], pair[0][0], pair[0][1], paths, num_threads),
            build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_1_converge"], pair[1][0], pair[1][1], paths, num_threads),
            build_pair_command(PAIR_STAGE_TO_SCRIPT["4_2_2"], pair, paths, num_threads),
            build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_2_converge"], pair[0][0], pair[0][1], paths, num_threads),
            build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_2_converge"], pair[1][0], pair[1][1], paths, num_threads),
            build_pair_command(PAIR_STAGE_TO_SCRIPT["4_2_3"], pair, paths, num_threads),
        ]
        run_commands(commands, dry_run=dry_run)
        return

    if stage == "full":
        agent, model = resolve_single_agent(args)
        pair = resolve_pair(args)
        commands = [
            build_core_command(CORE_STAGE_TO_SCRIPT[name], agent, model, paths, num_threads)
            for name in ("0", "1", "1_converge", "2", "2_1_converge", "2_2_converge")
        ]
        commands.extend(
            [
                build_pair_command(PAIR_STAGE_TO_SCRIPT["4_0"], pair, paths, num_threads),
                build_pair_command(PAIR_STAGE_TO_SCRIPT["4_1_1"], pair, paths, num_threads),
                build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_1_converge"], pair[0][0], pair[0][1], paths, num_threads),
                build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_1_converge"], pair[1][0], pair[1][1], paths, num_threads),
                build_pair_command(PAIR_STAGE_TO_SCRIPT["4_1_2"], pair, paths, num_threads),
                build_pair_command(PAIR_STAGE_TO_SCRIPT["4_2_1"], pair, paths, num_threads),
                build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_1_converge"], pair[0][0], pair[0][1], paths, num_threads),
                build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_1_converge"], pair[1][0], pair[1][1], paths, num_threads),
                build_pair_command(PAIR_STAGE_TO_SCRIPT["4_2_2"], pair, paths, num_threads),
                build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_2_converge"], pair[0][0], pair[0][1], paths, num_threads),
                build_single_stage_command(SINGLE_STAGE_TO_SCRIPT["4_2_2_converge"], pair[1][0], pair[1][1], paths, num_threads),
                build_pair_command(PAIR_STAGE_TO_SCRIPT["4_2_3"], pair, paths, num_threads),
            ]
        )
        run_commands(commands, dry_run=dry_run)
        return

    raise ValueError(f"Unsupported stage: {stage}")


if __name__ == "__main__":
    main()
