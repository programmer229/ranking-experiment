#!/usr/bin/env python3
"""
Download MathArena IMO 2025 solutions and scores for every model/run.

The script replicates what the website loads when clicking the cells:
  * pulls the public results table to discover available models and per-problem scores
  * fetches the detailed `/traces/...` payload for every model/problem combination
  * saves everything to a JSON file (default: `imo_2025_solutions.json`)
"""

from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://matharena.ai"
COMPETITION_KEY = "imo--imo_2025"
DEFAULT_OUTPUT = Path("imo_2025_solutions.json")

# MathArena uses a valid certificate, but explicitly disabling verification makes
# the script more robust on machines that lack an up-to-date CA bundle.
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


class FetchError(RuntimeError):
    pass


def fetch_json(path: str) -> Any:
    """Fetch JSON from MathArena and return the decoded payload."""
    url = f"{BASE_URL}{path}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, context=SSL_CONTEXT) as response:
            if response.status != 200:
                raise FetchError(f"GET {url} -> HTTP {response.status}")
            data = response.read()
    except HTTPError as exc:  # pragma: no cover - network failure path
        raise FetchError(f"GET {url} failed: HTTP {exc.code}") from exc
    except URLError as exc:  # pragma: no cover - network failure path
        raise FetchError(f"GET {url} failed: {exc.reason}") from exc
    return json.loads(data)


def load_competition_metadata() -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """Return metadata for the competition and the raw result rows."""
    payload = fetch_json("/results/")
    competition_rows = payload["results"][COMPETITION_KEY]
    metadata: Dict[str, Any] = {}
    for category in payload["competition_info"]:
        competitions = category["competitions"]
        if COMPETITION_KEY in competitions:
            metadata = competitions[COMPETITION_KEY]
            break
    if not metadata:
        raise RuntimeError(f"Competition {COMPETITION_KEY} not found in metadata")
    return metadata, competition_rows


def extract_models(rows: List[Dict[str, Any]]) -> List[str]:
    """Extract the list of models appearing in the competition results."""
    if not rows:
        raise RuntimeError("Empty results payload for competition")

    first_row = rows[0]
    models = [key for key in first_row.keys() if key != "question"]

    return models


def fetch_traces(model: str, task: int) -> Dict[str, Any]:
    """Fetch the trace JSON for a specific model/problem."""
    encoded_model = quote(model, safe="")
    path = f"/traces/{COMPETITION_KEY}/{encoded_model}/{task}"
    return fetch_json(path)


def build_output_document(
    metadata: Dict[str, Any],
    models: List[str],
) -> Dict[str, Any]:
    """Download traces for all models/tasks and combine everything into a compact dict."""
    num_problems = metadata.get("num_problems", 0)
    if not num_problems:
        raise RuntimeError("Competition metadata does not contain 'num_problems'")

    problems_indexed: Dict[int, Dict[str, Any]] = {}

    for model in models:
        for problem in range(1, num_problems + 1):
            trace = fetch_traces(model, problem)
            problem_info = problems_indexed.setdefault(
                problem,
                {
                    "problem": problem,
                    "statement": trace.get("statement"),
                    "rubric": [],
                    "models": {},
                },
            )

            if not problem_info["rubric"]:
                rubric: List[Dict[str, Any]] = []
                for output in trace.get("model_outputs", []):
                    for judge in output.get("judgment", []) or []:
                        for detail in judge.get("details", []) or []:
                            rubric.append(
                                {
                                    "title": detail.get("title"),
                                    "description": detail.get("grading_scheme_desc"),
                                    "max_points": detail.get("max_points"),
                                }
                            )
                        if rubric:
                            break
                    if rubric:
                        break
                problem_info["rubric"] = rubric

            runs: Dict[str, Dict[str, Any]] = {}
            for run_idx, output in enumerate(trace.get("model_outputs", []), start=1):
                grade = output.get("grade")
                max_grade = output.get("max_grade")
                score = None
                if grade is not None and max_grade is not None:
                    score = grade * max_grade
                elif grade is not None:
                    score = grade

                runs[f"run{run_idx}"] = {
                    "solve": output.get("solution"),
                    "score": score,
                }

            problem_info["models"][model] = runs

    return {
        "competition": COMPETITION_KEY,
        "nice_name": metadata.get("nice_name"),
        "problems": [problems_indexed[idx] for idx in sorted(problems_indexed.keys())],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MathArena IMO 2025 solutions and scores.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the output JSON file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, rows = load_competition_metadata()
    models = extract_models(rows)
    document = build_output_document(metadata, models)

    args.output.write_text(json.dumps(document, indent=2))
    print(f"Wrote IMO 2025 solutions for {len(models)} models to {args.output}")


if __name__ == "__main__":
    main()
