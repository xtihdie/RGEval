from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("data/300class/results")
DEFAULT_OUTPUT_SUBDIR = "3"
STEPS = ["0", "1", "2"]

VALID_LEVELS = {"初级", "中级", "专家级"}
REQUIRED_COLS = ["课堂ID", "维度", "人工标签", "分数", "AI评分", "评分理由"]

RERUN_STAGE = {
    "0": "0",
    "1_score": "1",
    "1_converge": "1_converge",
    "2_score": "2",
    "2_converge_1": "2_1_converge",
    "2_converge_2": "2_2_converge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Clean invalid result rows and emit rerun helpers")
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def has_valid_int_0_100(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for match in re.findall(r"\b(\d{1,3})\b", text):
        value = int(match)
        if 0 <= value <= 100:
            return True
    return False


def row_invalid(row: pd.Series) -> bool:
    try:
        int(row["课堂ID"])
        int(row["维度"])
    except Exception:
        return True

    if pd.isna(row["分数"]):
        return True

    if row["AI评分"] not in VALID_LEVELS:
        return True

    if not has_valid_int_0_100(str(row["评分理由"])):
        return True

    return False


def stage_from_filename(step: str, name: str) -> str:
    lowered = name.lower()
    if step == "0":
        return "score"
    if step == "1":
        return "converge" if "converge" in lowered else "score"
    if step == "2":
        if "2_1" in lowered or "converge_1" in lowered:
            return "converge_1"
        if "2_2" in lowered or "converge_2" in lowered:
            return "converge_2"
        return "score"
    raise ValueError(step)


def parse_cfg(filename: str) -> tuple[str, str, str]:
    parts = filename.replace(".csv", "").split("_")
    return parts[0], parts[1], parts[2]


def build_rerun_command(dataset_root: Path, stage_key: str, agent: str, model: str) -> str:
    stage = RERUN_STAGE[stage_key]
    return f'python main.py --dataset-root "{dataset_root}" --stage {stage} --agent "{agent}" --model "{model}"'


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    dataset_root = Path(args.dataset_root) if args.dataset_root else results_root.parent
    output_dir = Path(args.output_dir) if args.output_dir else results_root / DEFAULT_OUTPUT_SUBDIR

    bad_keys: dict[tuple[str, str, str, str], set[tuple[int, int]]] = defaultdict(set)
    bad_stage: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    print("Scanning step 0/1/2 outputs ...")

    for step in STEPS:
        stage_dir = results_root / step
        if not stage_dir.exists():
            continue

        for csv_path in stage_dir.glob("*.csv"):
            df = pd.read_csv(csv_path)

            if any(column not in df.columns for column in REQUIRED_COLS):
                print(f"Missing required columns, clearing file: {csv_path.name}")
                df.iloc[0:0].to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue

            stage = stage_from_filename(step, csv_path.name)
            _, agent, model = parse_cfg(csv_path.name)

            drop_indices: list[int] = []
            for index, row in df.iterrows():
                if row_invalid(row):
                    key = (int(row["课堂ID"]), int(row["维度"]))
                    bad_keys[(step, agent, model, stage)].add(key)
                    bad_stage[(step, agent, model)].add(stage)
                    drop_indices.append(index)

            if drop_indices:
                df.drop(index=drop_indices, inplace=True)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"Cleaned {len(drop_indices)} invalid rows from {csv_path.name}")

    def cascade(step: str, agent: str, model: str, target_stage: str) -> None:
        stage_dir = results_root / step
        for csv_path in stage_dir.glob("*.csv"):
            if stage_from_filename(step, csv_path.name) != target_stage:
                continue
            parsed_step, parsed_agent, parsed_model = parse_cfg(csv_path.name)
            if (parsed_step, parsed_agent, parsed_model) != (step, agent, model):
                continue

            df = pd.read_csv(csv_path)
            keys = set().union(
                *[value for key, value in bad_keys.items() if key[:3] == (step, agent, model)]
            )
            if not keys:
                continue

            mask = df.apply(
                lambda row: (int(row["课堂ID"]), int(row["维度"])) in keys,
                axis=1,
            )
            if mask.any():
                df = df.loc[~mask]
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                bad_stage[(step, agent, model)].add(target_stage)
                print(f"Cascaded cleanup to {csv_path.name}")

    for (step, agent, model), stages in list(bad_stage.items()):
        if step == "1" and "score" in stages:
            cascade("1", agent, model, "converge")
        if step == "2":
            if "score" in stages:
                cascade("2", agent, model, "converge_1")
                cascade("2", agent, model, "converge_2")
            if "converge_1" in stages:
                cascade("2", agent, model, "converge_2")

    commands: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    def add(stage_key: str, agent: str, model: str) -> None:
        marker = (stage_key, agent, model)
        if marker in seen:
            return
        seen.add(marker)
        commands.append(build_rerun_command(dataset_root, stage_key, agent, model))

    for (step, agent, model), stages in bad_stage.items():
        if step == "0":
            add("0", agent, model)
        elif step == "1":
            if "score" in stages:
                add("1_score", agent, model)
            if "converge" in stages:
                add("1_converge", agent, model)
        elif step == "2":
            if "score" in stages:
                add("2_score", agent, model)
            if "converge_1" in stages:
                add("2_converge_1", agent, model)
            if "converge_2" in stages:
                add("2_converge_2", agent, model)

    output_dir.mkdir(parents=True, exist_ok=True)

    sh_lines = ["#!/usr/bin/env bash", "set -e", "", "echo 'Rerun missing tasks'"]
    sh_lines.extend(commands)
    bat_lines = ["@echo off", "setlocal", "echo Rerun missing tasks"]
    bat_lines.extend(commands)

    sh_path = output_dir / "rerun_missing.sh"
    bat_path = output_dir / "rerun_missing.bat"
    sh_path.write_text("\n".join(sh_lines), encoding="utf-8")
    sh_path.chmod(0o755)
    bat_path.write_text("\n".join(bat_lines), encoding="utf-8")

    print(f"\nSaved rerun scripts: {sh_path} and {bat_path}")


if __name__ == "__main__":
    main()
