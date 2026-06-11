from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
TIMING_DIR = PROJECT_ROOT / "data" / "essay" / "results" / "timing"
ESSAY_FILE_PATH = PROJECT_ROOT / "data" / "essay" / "origin" / "ellipse_train_normalized.csv"

METHOD_STAGES = {
    "direct": ["essay_0"],
    "keyquestion": ["essay_2", "essay_2_converge"],
}
EVALUATE_PATH = PROJECT_ROOT / "tools" / "evaluate_essay_outputs.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run essay comparison experiments with timing logs")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--methods", nargs="+", choices=["direct", "keyquestion", "all"], default=["all"])
    parser.add_argument("--splits", nargs="+", default=["train", "test", "all"])
    parser.add_argument("--max-essays", type=int, default=None)
    parser.add_argument("--question-bank-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_methods(raw_methods: list[str]) -> list[str]:
    if "all" in raw_methods:
        return ["direct", "keyquestion"]
    return raw_methods


def build_stage_command(args: argparse.Namespace, stage: str, split: str) -> list[str]:
    command = [sys.executable, str(MAIN_PATH), "--dataset", "essay", "--stage", stage, "--agent", args.agent, "--model", args.model, "--split", split]
    if args.max_essays is not None:
        command.extend(["--max-essays", str(args.max_essays)])
    if args.question_bank_path:
        command.extend(["--question-bank-path", str(args.question_bank_path)])
    return command


def build_eval_command(args: argparse.Namespace, split: str) -> list[str]:
    return [sys.executable, str(EVALUATE_PATH), "--agent", args.agent, "--model", args.model, "--split", split]


def split_row_count(split: str) -> int:
    df = pd.read_csv(ESSAY_FILE_PATH, encoding="utf-8-sig")
    lowered = str(split).strip().lower()
    if lowered == "all":
        return len(df)
    return int((df["split"].astype(str).str.lower() == lowered).sum())


def main() -> None:
    args = parse_args()
    methods = resolved_methods(args.methods)
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = TIMING_DIR / f"essay_test_{args.agent}_{args.model}_{started_at}.json"

    run_log: dict[str, object] = {
        "agent": args.agent,
        "model": args.model,
        "methods": methods,
        "splits": args.splits,
        "max_essays": args.max_essays,
        "started_at": started_at,
        "stages": [],
    }

    total_start = time.perf_counter()
    for split in args.splits:
        available_rows = split_row_count(split)
        if available_rows == 0:
            run_log["stages"].append(
                {
                    "split": split,
                    "method": "availability",
                    "stage": "no_data",
                    "status": "skipped",
                    "returncode": 0,
                    "duration_seconds": 0.0,
                    "command": [],
                }
            )
            print(f"Skipping split={split}: no essay rows found in source data.")
            continue

        for method in methods:
            for stage in METHOD_STAGES[method]:
                command = build_stage_command(args, stage, split)
                pretty = " ".join(f'"{part}"' if " " in part else part for part in command)
                print(pretty)
                stage_started = time.perf_counter()
                status = "dry_run"
                returncode = 0
                if not args.dry_run:
                    completed = subprocess.run(command, cwd=PROJECT_ROOT)
                    returncode = completed.returncode
                    status = "ok" if returncode == 0 else "failed"
                    if returncode != 0:
                        run_log["stages"].append(
                            {
                                "split": split,
                                "method": method,
                                "stage": stage,
                                "status": status,
                                "returncode": returncode,
                                "duration_seconds": round(time.perf_counter() - stage_started, 3),
                                "command": command,
                            }
                        )
                        run_log["total_duration_seconds"] = round(time.perf_counter() - total_start, 3)
                        log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
                        raise SystemExit(returncode)
                run_log["stages"].append(
                    {
                        "split": split,
                        "method": method,
                        "stage": stage,
                        "status": status,
                        "returncode": returncode,
                        "duration_seconds": round(time.perf_counter() - stage_started, 3),
                        "command": command,
                    }
                )

        eval_command = build_eval_command(args, split)
        pretty = " ".join(f'"{part}"' if " " in part else part for part in eval_command)
        print(pretty)
        eval_started = time.perf_counter()
        eval_status = "dry_run"
        eval_returncode = 0
        if not args.dry_run:
            completed = subprocess.run(eval_command, cwd=PROJECT_ROOT)
            eval_returncode = completed.returncode
            eval_status = "ok" if eval_returncode == 0 else "failed"
            if eval_returncode != 0:
                run_log["stages"].append(
                    {
                        "split": split,
                        "method": "evaluation",
                        "stage": "metrics",
                        "status": eval_status,
                        "returncode": eval_returncode,
                        "duration_seconds": round(time.perf_counter() - eval_started, 3),
                        "command": eval_command,
                    }
                )
                run_log["total_duration_seconds"] = round(time.perf_counter() - total_start, 3)
                log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
                raise SystemExit(eval_returncode)
        run_log["stages"].append(
            {
                "split": split,
                "method": "evaluation",
                "stage": "metrics",
                "status": eval_status,
                "returncode": eval_returncode,
                "duration_seconds": round(time.perf_counter() - eval_started, 3),
                "command": eval_command,
            }
        )

    run_log["total_duration_seconds"] = round(time.perf_counter() - total_start, 3)
    log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Timing log written to: {log_path}")


if __name__ == "__main__":
    main()
