from __future__ import annotations

import argparse
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tqdm.auto import tqdm


DATASETS: Mapping[str, Path] = {
    "imo_2025": Path("imo_2025_solutions.json"),
}

DEFAULT_ELO_RATING = 1500.0


@dataclass
class RunCandidate:
    model: str
    run_id: str
    score: float
    solution: str


@dataclass
class Problem:
    problem_id: int
    statement: str
    runs: Sequence[RunCandidate]


class Judge:
    """Base class for deciding which run is better."""

    def compare(self, problem: Problem, run_a: RunCandidate, run_b: RunCandidate) -> int:
        """
        Return 1 if run_a is better, -1 if run_b is better, 0 if tied.
        """
        raise NotImplementedError


class ScoreJudge(Judge):
    """Fallback judge that compares runs using their numeric score."""

    def compare(self, problem: Problem, run_a: RunCandidate, run_b: RunCandidate) -> int:
        if math.isclose(run_a.score, run_b.score):
            return 0
        return 1 if run_a.score > run_b.score else -1


class LLMJudge(Judge):
    """
    Judge that queries an LLM to compare two solutions.

    Requires the OpenAI Python package and a valid `OPENAI_API_KEY`.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_workers: int = 4) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai package not installed. Install `openai` to use the LLM judge."
            ) from exc

        api_key = Path(".openai-key").read_text().strip() if Path(".openai-key").exists() else None
        if not api_key:
            import os

            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "OPENAI_API_KEY not set and .openai-key file not found; cannot use LLM judge."
            )

        self._client = OpenAI(api_key=api_key)  # type: ignore
        self._model = model
        self._temperature = temperature
        self._max_workers = max(1, max_workers)

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def compare(self, problem: Problem, run_a: RunCandidate, run_b: RunCandidate) -> int:
        prompt = (
            "You are grading solutions to an IMO problem.\n"
            "Problem statement:\n"
            f"{problem.statement}\n\n"
            "Compare the two solutions below and reply with a single character:\n"
            "'A' if Solution A is better, 'B' if Solution B is better, or 'T' if they are tied.\n\n"
            f"Solution A:\n"
            f"{run_a.solution}\n\n"
            f"Solution B:\n"
            f"{run_b.solution}\n"
        )
        response = self._client.responses.create(  # type: ignore
            model=self._model,
            input=prompt,
            #temperature=self._temperature,
            max_output_tokens=32,
        )
        try:
            content = response.output[0].content[0].text.strip().upper()  # type: ignore[attr-defined]
        except (AttributeError, IndexError, ValueError):  # pragma: no cover - API fallback
            content = getattr(response, "output_text", "").strip().upper()
        if content.startswith("A"):
            return 1
        if content.startswith("B"):
            return -1
        return 0


def compute_pairwise_results(
    problem: Problem,
    candidates: Sequence[RunCandidate],
    judge: Judge,
    desc: str,
) -> Dict[Tuple[int, int], int]:
    pairs = list(itertools.combinations(range(len(candidates)), 2))
    results: Dict[Tuple[int, int], int] = {}
    if not pairs:
        return results

    if isinstance(judge, LLMJudge):
        with ThreadPoolExecutor(max_workers=judge.max_workers) as executor:
            future_to_pair = {
                executor.submit(judge.compare, problem, candidates[i], candidates[j]): (i, j)
                for i, j in pairs
            }
            for future in tqdm(as_completed(future_to_pair), total=len(future_to_pair), desc=desc):
                pair = future_to_pair[future]
                verdict = future.result()
                results[pair] = verdict
    else:
        for pair in tqdm(pairs, desc=desc):
            i, j = pair
            results[pair] = judge.compare(problem, candidates[i], candidates[j])

    return results


def load_dataset(dataset_key: str) -> List[Problem]:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {', '.join(DATASETS)}")

    path = DATASETS[dataset_key]
    raw = json.loads(path.read_text())

    problems: List[Problem] = []
    for problem_entry in raw["problems"]:
        candidates: List[RunCandidate] = []
        for model_name, runs in problem_entry["models"].items():
            for run_id, payload in sorted(runs.items()):
                score = payload.get("score")
                candidates.append(
                    RunCandidate(
                        model=model_name,
                        run_id=run_id,
                        score=float(score) if score is not None else 0.0,
                        solution=payload.get("solve", ""),
                    )
                )
        problems.append(
            Problem(
                problem_id=int(problem_entry["problem"]),
                statement=problem_entry["statement"],
                runs=candidates,
            )
        )
    return problems


def pairwise_rank_runs(
    problem: Problem,
    candidates: Sequence[RunCandidate],
    judge: Judge,
) -> Tuple[List[int], Dict[int, float]]:
    if not candidates:
        return [], {}

    wins: Dict[int, float] = {idx: 0.0 for idx in range(len(candidates))}
    pair_results = compute_pairwise_results(
        problem,
        candidates,
        judge,
        desc=f"Comparisons P{problem.problem_id}",
    )
    for (i, j), verdict in pair_results.items():
        if verdict == 0:
            wins[i] += 0.5
            wins[j] += 0.5
        elif verdict > 0:
            wins[i] += 1.0
        else:
            wins[j] += 1.0

    ordering = sorted(
        range(len(candidates)),
        key=lambda idx: (wins[idx], candidates[idx].score),
        reverse=True,
    )
    return ordering, wins


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_rank_runs(
    candidates: Sequence[RunCandidate],
    judge: Judge,
    problem: Problem,
    k_factor: float,
) -> Tuple[List[int], Dict[int, float]]:
    if not candidates:
        return [], {}

    ratings: Dict[int, float] = {idx: DEFAULT_ELO_RATING for idx in range(len(candidates))}
    pair_results = compute_pairwise_results(
        problem,
        candidates,
        judge,
        desc=f"Elo comparisons P{problem.problem_id}",
    )

    for (i, j), verdict in pair_results.items():
        if verdict == 0:
            outcome_i = 0.5
        elif verdict > 0:
            outcome_i = 1.0
        else:
            outcome_i = 0.0

        expected_i = elo_expected_score(ratings[i], ratings[j])
        expected_j = 1.0 - expected_i

        ratings[i] += k_factor * (outcome_i - expected_i)
        ratings[j] += k_factor * ((1.0 - outcome_i) - expected_j)

    ordering = sorted(
        range(len(candidates)),
        key=lambda idx: ratings[idx],
        reverse=True,
    )
    return ordering, ratings


def format_pairwise_table(
    problem: Problem,
    ordering: Sequence[int],
    wins: Mapping[int, float],
) -> str:
    header = f"Problem {problem.problem_id} Pairwise Rankings"
    subheader = f"{'Rank':<4} {'Model':<25} {'Run':<6} {'Wins':>6} {'Score':>7}"
    lines = [header, subheader, "-" * len(subheader)]
    for rank, idx in enumerate(ordering, start=1):
        run = problem.runs[idx]
        lines.append(
            f"{rank:<4} {run.model:<25} {run.run_id:<6} {wins[idx]:>6.1f} {run.score:>7.2f}"
        )
    return "\n".join(lines)


def format_elo_table(
    problem: Problem,
    ordering: Sequence[int],
    ratings: Mapping[int, float],
) -> str:
    header = f"Problem {problem.problem_id} Elo Rankings"
    subheader = f"{'Rank':<4} {'Model':<25} {'Run':<6} {'Rating':>8} {'Score':>7}"
    lines = [header, subheader, "-" * len(subheader)]
    for rank, idx in enumerate(ordering, start=1):
        run = problem.runs[idx]
        lines.append(
            f"{rank:<4} {run.model:<25} {run.run_id:<6} {ratings[idx]:>8.1f} {run.score:>7.2f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank math competition models using different schemes.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        default="imo_2025",
        help="Dataset to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--ranking-scheme",
        choices=("pairwise", "elo"),
        default="pairwise",
        help="Ranking scheme to run (default: %(default)s).",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="Elo K-factor (only relevant when --ranking-scheme=elo).",
    )
    parser.add_argument(
        "--judge",
        choices=("score", "llm"),
        default="llm",
        help="Judging strategy for pairwise comparisons (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model to use when --judge=llm.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=4,
        help="Maximum parallel requests when using the LLM judge (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the ranked runs JSON artifact.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional path to save per-problem summary statistics JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems = load_dataset(args.dataset)

    if args.judge == "llm":
        judge: Judge = LLMJudge(args.llm_model, max_workers=args.llm_workers)
    else:
        judge = ScoreJudge()

    artifact: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for problem in problems:
        if len(problem.runs) <= 1:
            ordering = list(range(len(problem.runs)))
            wins: Dict[int, float] = {idx: 0.0 for idx in ordering}
            ratings: Dict[int, float] = {idx: DEFAULT_ELO_RATING for idx in ordering}
        elif args.ranking_scheme == "pairwise":
            ordering, wins = pairwise_rank_runs(problem, problem.runs, judge)
            ratings = {idx: None for idx in ordering}
        else:
            ordering, ratings = elo_rank_runs(problem.runs, judge, problem, args.k_factor)
            wins = {idx: None for idx in ordering}

        if args.ranking_scheme == "pairwise":
            print(format_pairwise_table(problem, ordering, wins))
        else:
            print(format_elo_table(problem, ordering, ratings))

        ranking_records: List[Dict[str, Any]] = []
        for position, idx in enumerate(ordering, start=1):
            run = problem.runs[idx]
            record: Dict[str, Any] = {
                "rank": position,
                "model": run.model,
                "run": run.run_id,
                "score": run.score,
                "solution": run.solution,
            }
            if args.ranking_scheme == "pairwise":
                record["wins"] = wins.get(idx)
            else:
                record["rating"] = ratings.get(idx)
            ranking_records.append(record)

        scores_all = [run.score for run in problem.runs]
        total_score = sum(scores_all)
        max_score = max(scores_all, default=0.0)
        rank_sum_actual = sum(rec["rank"] for rec in ranking_records)
        rank_max = ranking_records[-1]["rank"] if ranking_records else 0
        num_runs = len(problem.runs)
        weighted_score_sum = sum(
            rec["score"] / rec["rank"] for rec in ranking_records if rec["rank"] > 0
        )

        sorted_scores_desc = sorted(scores_all, reverse=True)
        weighted_score_sum_optimal = sum(
            score / idx for idx, score in enumerate(sorted_scores_desc, start=1)
        )
        rank_sum_optimal = sum(range(1, num_runs + 1)) if num_runs else 0

        artifact.append(
            {
                "problem": problem.problem_id,
                "statement": problem.statement,
                "ranking_scheme": args.ranking_scheme,
                "k_factor": args.k_factor if args.ranking_scheme == "elo" else None,
                "ranked_runs": ranking_records,
            }
        )

        print()

        summary.append(
            {
                "problem": problem.problem_id,
                "ranking_scheme": args.ranking_scheme,
                "judge": args.judge,
                "total_runs": len(problem.runs),
                "score_sum": total_score,
                "score_sum_optimal": sum(sorted_scores_desc),
                "score_max": max_score,
                "rank_sum": rank_sum_actual,
                "rank_sum_optimal": rank_sum_optimal,
                "rank_max": rank_max,
                "weighted_score_sum": weighted_score_sum,
                "weighted_score_sum_optimal": weighted_score_sum_optimal,
            }
        )

        print(
            f"Weighted score sum (actual / optimal): "
            f"{weighted_score_sum:.4f} / {weighted_score_sum_optimal:.4f}"
        )
        print()

    if args.output:
        args.output.write_text(json.dumps(artifact, indent=2))
        print(f"Saved rankings to {args.output}")

    if args.summary_output:
        args.summary_output.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {args.summary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
