from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMING_DIR = PROJECT_ROOT / "data" / "wiki_quality" / "results" / "timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize total wiki test time from timing logs")
    parser.add_argument("--log", type=str, default=None, help="Optional single timing log path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = [Path(args.log)] if args.log else sorted(DEFAULT_TIMING_DIR.glob("wiki_test_*.json"))
    if not logs:
        raise FileNotFoundError(f"No timing logs found in {DEFAULT_TIMING_DIR}")

    total = 0.0
    for log in logs:
        payload = json.loads(log.read_text(encoding="utf-8"))
        duration = float(payload.get("total_duration_seconds", 0.0))
        total += duration
        print(f"{log.name}: {duration:.3f}s")
        for stage in payload.get("stages", []):
            split = stage.get("split", "n/a")
            print(f"  {split} | {stage['method']}/{stage['stage']}: {float(stage['duration_seconds']):.3f}s [{stage['status']}]")
    print(f"Total accumulated test time: {total:.3f}s")


if __name__ == "__main__":
    main()
