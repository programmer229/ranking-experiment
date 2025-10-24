#!/usr/bin/env python3
"""
Create a down-sampled version of `imo_2025_solutions.json` containing only a
limited number of runs per problem (default: 8).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT = Path("imo_2025_solutions.json")
DEFAULT_OUTPUT = Path("imo_2025_solutions_sampled.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a smaller set of runs per problem.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the solutions JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the sampled JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=8,
        help="Number of runs to keep per problem (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling (default: 0).",
    )
    return parser.parse_args()


def sample_problem(problem: Dict[str, Any], num_runs: int, rng: random.Random) -> Dict[str, Any]:
    """Return a dictionary with at most `num_runs` runs for the given problem."""
    all_runs: List[Tuple[str, str]] = []
    for model, runs in problem["models"].items():
        for run_id in runs.keys():
            all_runs.append((model, run_id))

    if len(all_runs) <= num_runs:
        return problem

    selected = set(rng.sample(all_runs, num_runs))
    sampled_models: Dict[str, Dict[str, Any]] = {}
    for model, runs in problem["models"].items():
        filtered_runs = {
            run_id: data for run_id, data in runs.items() if (model, run_id) in selected
        }
        if filtered_runs:
            sampled_models[model] = filtered_runs

    sampled_problem = dict(problem)
    sampled_problem["models"] = sampled_models
    sampled_problem["sample_size"] = len(selected)
    return sampled_problem


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data = json.loads(args.input.read_text())
    sampled = dict(data)

    sampled_problems: List[Dict[str, Any]] = []
    for problem in data.get("problems", []):
        sampled_problems.append(sample_problem(problem, args.num_runs, rng))

    sampled["problems"] = sampled_problems
    args.output.write_text(json.dumps(sampled, indent=2))
    print(f"Wrote sampled solutions to {args.output}")


if __name__ == "__main__":
    main()
