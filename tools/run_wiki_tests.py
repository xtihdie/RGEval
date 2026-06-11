from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
TIMING_DIR = PROJECT_ROOT / "data" / "wiki_quality" / "results" / "timing"

METHOD_STAGES = {
    "direct": ["wiki_0"],
    "keyquestion": ["wiki_2", "wiki_2_converge"],
}
EVALUATE_PATH = PROJECT_ROOT / "tools" / "evaluate_wiki_outputs.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run wiki comparison experiments with timing logs")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--methods", nargs="+", choices=["direct", "keyquestion", "all"], default=["all"])
    parser.add_argument("--splits", nargs="+", default=["train", "test", "all"])
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--question-bank-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_methods(raw_methods: list[str]) -> list[str]:
    if "all" in raw_methods:
        return ["direct", "keyquestion"]
    return raw_methods


def build_stage_command(args: argparse.Namespace, stage: str, split: str) -> list[str]:
    command = [sys.executable, str(MAIN_PATH), "--dataset", "wiki_quality", "--stage", stage, "--agent", args.agent, "--model", args.model, "--split", split]
    if args.max_articles is not None:
        command.extend(["--max-essays", str(args.max_articles)])
    if args.question_bank_path:
        command.extend(["--question-bank-path", str(args.question_bank_path)])
    return command


def build_eval_command(args: argparse.Namespace, split: str) -> list[str]:
    return [sys.executable, str(EVALUATE_PATH), "--agent", args.agent, "--model", args.model, "--split", split]


def main() -> None:
    args = parse_args()
    methods = resolved_methods(args.methods)
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = TIMING_DIR / f"wiki_test_{args.agent}_{args.model}_{started_at}.json"

    run_log: dict[str, object] = {
        "agent": args.agent,
        "model": args.model,
        "methods": methods,
        "splits": args.splits,
        "max_articles": args.max_articles,
        "started_at": started_at,
        "stages": [],
    }

    total_start = time.perf_counter()
    for split in args.splits:
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
