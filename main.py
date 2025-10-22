from __future__ import annotations

import argparse
import itertools
import json
import math
import random
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
    rubric: Sequence[Dict[str, Any]]
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
        rubric_section = ""
        if problem.rubric:
            rubric_lines = []
            for idx, item in enumerate(problem.rubric, start=1):
                title = item.get("title") or ""
                desc = item.get("description") or ""
                max_points = item.get("max_points")
                max_info = f" (max {max_points})" if max_points is not None else ""
                rubric_lines.append(f"{idx}. {title}{max_info}: {desc}")
            rubric_section = "Rubric:\n" + "\n".join(rubric_lines) + "\n\n"
        prompt = (
            "You are grading solutions to an IMO problem.\n"
            "Problem statement:\n"
            f"{problem.statement}\n\n"
            f"{rubric_section}"
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
            #max_output_tokens=32,
        )
        content_text = ""
        output = getattr(response, "output", None)
        if output:
            try:
                first_item = output[0]
                content_blocks = getattr(first_item, "content", None)
                if content_blocks:
                    content_text = (content_blocks[0].text or "").strip()
            except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
                content_text = ""
        if not content_text:
            content_text = getattr(response, "output_text", "")
        content = content_text.strip().upper()
        #print(content)
        if content.startswith("A"):
            return 1
        if content.startswith("B"):
            return -1
        return 0


class LLMRubricScorer:
    """Scores individual solutions using the rubric via an LLM."""

    def __init__(self, model: str, temperature: float = 0.0, max_workers: int = 4) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. Install `openai` to use the LLM judge."
            ) from exc

        api_key = Path(".openai-key").read_text().strip() if Path(".openai-key").exists() else None
        if not api_key:
            import os

            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:  # pragma: no cover
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

    def score(self, problem: Problem, run: RunCandidate) -> float:
        rubric_lines = []
        total_points = 0.0
        for item in problem.rubric:
            title = item.get("title") or ""
            desc = item.get("description") or ""
            max_points = item.get("max_points")
            if max_points is None:
                max_points = 1.0
            total_points += max_points
            rubric_lines.append(f"- {title} (max {max_points}): {desc}")
        rubric_section = ""
        if rubric_lines:
            rubric_section = "Rubric:\n" + "\n".join(rubric_lines) + "\n\n"

        prompt = (
            "You are grading a solution to an IMO problem.\n"
            "Evaluate the solution strictly according to the rubric and reply ONLY with a JSON object "
            'containing a float field "total_score" and an array "per_item" with entries '
            'that include "title", "awarded", "max", and "reason".\n'
            f"The maximum total score is {total_points}. Partial credit is allowed.\n\n"
            f"Problem statement:\n{problem.statement}\n\n"
            f"{rubric_section}"
            f"Solution:\n{run.solution}\n"
        )

        response = self._client.responses.create(  # type: ignore
            model=self._model,
            input=prompt,
            temperature=self._temperature,
            #max_output_tokens=256,
        )

        content_text = ""
        output = getattr(response, "output", None)
        if output:
            try:
                first_item = output[0]
                content_blocks = getattr(first_item, "content", None)
                if content_blocks:
                    content_text = (content_blocks[0].text or "").strip()
            except (AttributeError, IndexError, TypeError):
                content_text = ""
        if not content_text:
            content_text = getattr(response, "output_text", "")

        json_text = content_text.strip()
        start = json_text.find("{")
        end = json_text.rfind("}")
        if start != -1 and end != -1:
            json_text = json_text[start : end + 1]
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return 0.0

        total_score = data.get("total_score", 0.0)
        try:
            score = float(total_score)
        except (TypeError, ValueError):
            score = 0.0

        if total_points > 0:
            score = max(0.0, min(score, total_points))
        return score


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


def score_with_llm(
    problem: Problem,
    candidates: Sequence[RunCandidate],
    scorer: LLMRubricScorer,
) -> Dict[int, float]:
    if not candidates:
        return {}

    results: Dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=scorer.max_workers) as executor:
        futures = {
            executor.submit(scorer.score, problem, candidates[idx]): idx
            for idx in range(len(candidates))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Grading P{problem.problem_id}"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = 0.0
    return results


def load_dataset(dataset_key: str, custom_path: Path | None = None) -> List[Problem]:
    if custom_path is not None:
        path = custom_path
    else:
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
                rubric=problem_entry.get("rubric", []),
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


def score_rank_runs(
    candidates: Sequence[RunCandidate],
    rng: random.Random,
) -> Tuple[List[int], Dict[int, float]]:
    if not candidates:
        return [], {}

    scores: Dict[int, float] = {idx: candidates[idx].score if candidates[idx].score is not None else 0.0 for idx in range(len(candidates))}
    ordering = list(range(len(candidates)))
    rng.shuffle(ordering)
    ordering.sort(key=lambda idx: scores[idx], reverse=True)
    return ordering, scores


def rank_indices_by_metric(
    metric_map: Mapping[int, float],
    rng: random.Random,
) -> List[int]:
    ordering = list(metric_map.keys())
    rng.shuffle(ordering)
    ordering.sort(key=lambda idx: metric_map[idx], reverse=True)
    return ordering


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


def format_score_table(
    problem: Problem,
    ordering: Sequence[int],
    scores: Mapping[int, float],
    metric_scores: Mapping[int, float] | None = None,
    metric_label: str = "Rank Metric",
) -> str:
    if metric_scores is not None:
        header = f"Problem {problem.problem_id} Score Rankings"
        subheader = f"{'Rank':<4} {'Model':<25} {'Run':<6} {'Score':>7} {metric_label:>14}"
        lines = [header, subheader, "-" * len(subheader)]
        for rank, idx in enumerate(ordering, start=1):
            run = problem.runs[idx]
            lines.append(
                f"{rank:<4} {run.model:<25} {run.run_id:<6} {scores[idx]:>7.2f} {metric_scores[idx]:>14.2f}"
            )
    else:
        header = f"Problem {problem.problem_id} Score Rankings"
        subheader = f"{'Rank':<4} {'Model':<25} {'Run':<6} {'Score':>7}"
        lines = [header, subheader, "-" * len(subheader)]
        for rank, idx in enumerate(ordering, start=1):
            run = problem.runs[idx]
            lines.append(
                f"{rank:<4} {run.model:<25} {run.run_id:<6} {scores[idx]:>7.2f}"
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
        choices=("pairwise", "elo", "score", "llm_score"),
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
        "--solutions-json",
        type=Path,
        help="Override the dataset file path (e.g. use a sampled JSON).",
    )
    parser.add_argument(
        "--score-seed",
        type=int,
        default=0,
        help="Seed for tie-breaking in score-based ranking (default: %(default)s).",
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
    problems = load_dataset(args.dataset, args.solutions_json)

    judge: Judge | None = None
    scorer: LLMRubricScorer | None = None
    if args.ranking_scheme in {"pairwise", "elo"}:
        if args.judge == "llm":
            judge = LLMJudge(args.llm_model, max_workers=args.llm_workers)
        else:
            judge = ScoreJudge()
    if args.ranking_scheme == "llm_score":
        scorer = LLMRubricScorer(args.llm_model, max_workers=args.llm_workers)

    artifact: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    score_rng = random.Random(args.score_seed)

    for problem in problems:
        if args.ranking_scheme == "pairwise":
            if judge is None:
                raise RuntimeError("Pairwise ranking requires a judge.")
            ordering, metric_map = pairwise_rank_runs(problem, problem.runs, judge)
            print(format_pairwise_table(problem, ordering, metric_map))
        elif args.ranking_scheme == "elo":
            if judge is None:
                raise RuntimeError("Elo ranking requires a judge.")
            ordering, metric_map = elo_rank_runs(problem.runs, judge, problem, args.k_factor)
            print(format_elo_table(problem, ordering, metric_map))
        elif args.ranking_scheme == "score":
            ordering, metric_map = score_rank_runs(problem.runs, score_rng)
            print(format_score_table(problem, ordering, metric_map, None))
        elif args.ranking_scheme == "llm_score":
            if scorer is None:
                raise RuntimeError("LLM scoring requires an LLM rubric scorer.")
            metric_map = score_with_llm(problem, problem.runs, scorer)
            ordering = rank_indices_by_metric(metric_map, score_rng)
            original_scores = {
                idx: problem.runs[idx].score if problem.runs[idx].score is not None else 0.0
                for idx in range(len(problem.runs))
            }
            print(format_score_table(problem, ordering, original_scores, metric_map, metric_label="LLM Score"))
        else:
            raise ValueError(f"Unsupported ranking scheme {args.ranking_scheme}")

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
            metric_value = metric_map.get(idx) if metric_map else None
            if args.ranking_scheme == "pairwise":
                record["wins"] = metric_map.get(idx)
            elif args.ranking_scheme == "elo":
                record["rating"] = metric_map.get(idx)
            elif args.ranking_scheme == "score":
                record["score_rank_score"] = metric_map.get(idx)
            elif args.ranking_scheme == "llm_score":
                record["llm_score"] = metric_map.get(idx)
            if metric_value is not None:
                record["rank_metric"] = metric_value
            ranking_records.append(record)

        scores_all = [run.score for run in problem.runs]
        total_score = sum(scores_all)
        max_score = max(scores_all, default=0.0)
        rank_sum_actual = sum(rec["rank"] for rec in ranking_records)
        rank_max = ranking_records[-1]["rank"] if ranking_records else 0
        num_runs = len(problem.runs)

        indices = list(range(num_runs))
        score_values = {
            idx: scores_all[idx] if scores_all[idx] is not None else 0.0 for idx in indices
        }
        metric_values = {
            idx: metric_map.get(idx, score_values[idx]) if metric_map else score_values[idx]
            for idx in indices
        }

        harmonic_sum = sum(1.0 / k for k in range(1, num_runs + 1)) if num_runs else 0.0
        harmonic_dcg_sum = (
            sum(1.0 / math.log2(k + 1) for k in range(1, num_runs + 1)) if num_runs else 0.0
        )

        def weighted_sum(order: Sequence[int], values: Mapping[int, float]) -> float:
            return sum(
                values[idx] / rank for rank, idx in enumerate(order, start=1)
            )

        def dcg(order: Sequence[int], values: Mapping[int, float]) -> float:
            return sum(
                values[idx] / math.log2(rank + 1)
                for rank, idx in enumerate(order, start=1)
            )

        def optimal_order(values: Mapping[int, float]) -> List[int]:
            return sorted(values.keys(), key=lambda idx: (values[idx], -idx), reverse=True)

        avg_score = total_score / num_runs if num_runs else 0.0
        score_order_optimal = optimal_order(score_values)
        weighted_score_sum_actual = weighted_sum(ordering, score_values)
        weighted_score_sum_optimal = weighted_sum(score_order_optimal, score_values)
        weighted_score_sum_random = avg_score * harmonic_sum

        score_dcg_actual = dcg(ordering, score_values)
        score_dcg_optimal = dcg(score_order_optimal, score_values)
        score_dcg_random = avg_score * harmonic_dcg_sum
        score_ndcg_actual = (
            score_dcg_actual / score_dcg_optimal if score_dcg_optimal > 0 else 0.0
        )
        score_ndcg_random = (
            score_dcg_random / score_dcg_optimal if score_dcg_optimal > 0 else 0.0
        )

        avg_metric = (
            sum(metric_values.values()) / num_runs if num_runs else 0.0
        )
        metric_order_optimal = optimal_order(metric_values)
        weighted_metric_sum_actual = weighted_sum(ordering, metric_values)
        weighted_metric_sum_optimal = weighted_sum(metric_order_optimal, metric_values)
        weighted_metric_sum_random = avg_metric * harmonic_sum

        metric_dcg_actual = dcg(ordering, metric_values)
        metric_dcg_optimal = dcg(metric_order_optimal, metric_values)
        metric_dcg_random = avg_metric * harmonic_dcg_sum
        metric_ndcg_actual = (
            metric_dcg_actual / metric_dcg_optimal if metric_dcg_optimal > 0 else 0.0
        )
        metric_ndcg_random = (
            metric_dcg_random / metric_dcg_optimal if metric_dcg_optimal > 0 else 0.0
        )

        sorted_scores_desc = sorted(score_values.values(), reverse=True)
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
                "weighted_score_sum_actual": weighted_score_sum_actual,
                "weighted_score_sum_optimal": weighted_score_sum_optimal,
                "weighted_score_sum_random": weighted_score_sum_random,
                "weighted_metric_sum_actual": weighted_metric_sum_actual,
                "weighted_metric_sum_optimal": weighted_metric_sum_optimal,
                "weighted_metric_sum_random": weighted_metric_sum_random,
                "score_dcg_actual": score_dcg_actual,
                "score_dcg_optimal": score_dcg_optimal,
                "score_dcg_random": score_dcg_random,
                "score_ndcg_actual": score_ndcg_actual,
                "score_ndcg_random": score_ndcg_random,
                "metric_dcg_actual": metric_dcg_actual,
                "metric_dcg_optimal": metric_dcg_optimal,
                "metric_dcg_random": metric_dcg_random,
                "metric_ndcg_actual": metric_ndcg_actual,
                "metric_ndcg_random": metric_ndcg_random,
            }
        )

        print(
            f"Weighted score sum (rubric actual / optimal / random ~): "
            f"{weighted_score_sum_actual:.4f} / {weighted_score_sum_optimal:.4f} / {weighted_score_sum_random:.4f}"
        )
        print(
            f"Weighted metric sum (rank metric actual / optimal / random ~): "
            f"{weighted_metric_sum_actual:.4f} / {weighted_metric_sum_optimal:.4f} / {weighted_metric_sum_random:.4f}"
        )
        print(
            f"NDCG score-based (actual / random ~): {score_ndcg_actual:.4f} / {score_ndcg_random:.4f}"
        )
        print(
            f"NDCG metric-based (actual / random ~): {metric_ndcg_actual:.4f} / {metric_ndcg_random:.4f}"
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
