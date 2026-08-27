# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create training and reward-preserving dual-Judge benchmark configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = (
    ("recorded", "recorded", "terminal_once"),
    ("accuracy", "accuracy", "terminal_once"),
    ("accuracy_shadow", "accuracy_shadow", "terminal_once"),
    ("dual", "dual", "terminal_once"),
    ("dual_shadow", "dual_shadow", "terminal_once"),
    ("dual_per_turn", "dual", "per_turn"),
    ("dual_shadow_per_turn", "dual_shadow", "per_turn"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name("judge_services.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, benchmark_mode, reasoning_trigger in VARIANTS:
        config = dict(base)
        config["benchmark_mode"] = benchmark_mode
        config["reasoning_trigger"] = reasoning_trigger
        output = args.output_dir / f"judge_services_{name}.json"
        output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
