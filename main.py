from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm


DATASETS: Dict[str, Path] = {
    "imo_2025": Path("imo_2025_solutions.json"),
    "creative_writing": Path("creative_writing_dataset.json"),
}

DEFAULT_ELO_RATING = 1500.0

IMO_COMPARE_GUIDELINES = textwrap.dedent(
    """\
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify each provided mathematical solution exactly as written. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.
### Instructions ###
**1. Core Instructions**
* Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
* You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.
**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.
* **a. Critical Error:** This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that ‘A > B, C > D’ implies ‘A - C > B - D’) and **factual errors** (e.g., a calculation error like ‘2 + 3 = 6’).
  * **Procedure:** Explain the specific error and state that it **invalidates the current line of reasoning**. Do NOT check any further steps that rely on this error. You MUST, however, scan the rest of the solution to identify and verify any fully independent parts (such as disjoint cases).
* **b. Justification Gap:** This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
  * **Procedure:** Explain the gap in the justification. State that you will **assume the step’s conclusion is true** for the sake of argument, and then proceed to verify all subsequent steps to check if the remainder of the argument is sound.
### Output Format ###
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.
**a. Summary**
This section MUST be at the very beginning of your response. It must contain two components:
* **Final Verdict:** A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution’s approach is viable but contains several Justification Gaps."
* **List of Findings:** A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
  * **Location:** A direct quote of the key phrase or equation where the issue occurs.
  * **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).
**b. Detailed Verification Log**
Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.
### Example of the Required Summary Format ###
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*
**Final Verdict:** The solution is **invalid** because it contains a Critical Error.
**List of Findings:**
* **Location:** "By interchanging the limit and the integral, we get ..."
* **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
* **Location:** "From $A > B$ and $C > D$, it follows that $A - C > B - D$"
* **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.
"""
).strip()

CREATIVE_COMPARE_GUIDELINES = textwrap.dedent(
    """\
You are an award-winning literary editor evaluating two short stories written in response to the same prompt. Judge each piece holistically, balancing creativity with technical execution. Consider:
* Alignment with the prompt’s theme or constraints.
* Originality of concept and strength of the narrative voice.
* Structure, pacing, clarity, and cohesion of the storytelling.
* Character or imagery development and the emotional journey offered to the reader.
* Command of language, stylistic control, and polish.
Identify strengths and weaknesses in both pieces before deciding which one is superior. Only declare a tie if the stories are nearly indistinguishable in overall quality or excel in complementary ways.""",
).strip()

IMO_SCORE_GUIDELINES = textwrap.dedent(
    """\
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify each provided mathematical solution exactly as written. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.
### Instructions ###
**1. Core Instructions**
* Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
* You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.
**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.
* **a. Critical Error:** This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that ‘A > B, C > D’ implies ‘A - C > B - D’) and **factual errors** (e.g., a calculation error like ‘2 + 3 = 6’).
  * **Procedure:** Explain the specific error and state that it **invalidates the current line of reasoning**. Do NOT check any further steps that rely on this error. You MUST, however, scan the rest of the solution to identify and verify any fully independent parts (such as disjoint cases).
* **b. Justification Gap:** This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
  * **Procedure:** Explain the gap in the justification. State that you will **assume the step’s conclusion is true** for the sake of argument, and then proceed to verify all subsequent steps to check if the remainder of the argument is sound.
### Output Format ###
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.
**a. Summary**
This section MUST be at the very beginning of your response. It must contain two components:
* **Final Verdict:** A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution’s approach is viable but contains several Justification Gaps."
* **List of Findings:** A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
  * **Location:** A direct quote of the key phrase or equation where the issue occurs.
  * **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).
**b. Detailed Verification Log**
Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.
### Example of the Required Summary Format ###
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*
**Final Verdict:** The solution is **invalid** because it contains a Critical Error.
**List of Findings:**
* **Location:** "By interchanging the limit and the integral, we get ..."
* **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
* **Location:** "From $A > B$ and $C > D$, it follows that $A - C > B - D$"
* **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.
### Rubric Scoring ###
After completing the verification workflow, evaluate each rubric item. Award points between 0 and the specified maximum, granting partial credit only when the reasoning in the solution supports it. A rubric item that contains a Critical Error must receive 0 for the affected portion; a Justification Gap may receive partial credit if you explain why the remaining reasoning still supports some credit.
### Response Format ###
For this scoring interface, you must still perform the full analysis above, but you must NOT output the Summary or Detailed Verification Log. After you finish grading, respond **only** with a JSON object containing:
* **"total_score"** — the sum of the awarded points (float).
* **"per_item"** — an array of objects, one per rubric item, each with:
  * **"title"** — the rubric item title.
  * **"awarded"** — the points you are awarding (float).
  * **"max"** — the maximum points available for that item (float).
  * **"reason"** — a concise explanation referencing the findings that justify the awarded score.
Ensure that "total_score" equals the sum of the "awarded" values.
"""
).strip()

CREATIVE_SCORE_GUIDELINES = textwrap.dedent(
    """\
You are an award-winning literary editor providing a holistic evaluation of a short story written for the specified prompt. Score the piece on the following dimensions, awarding fractional credit as needed:
1. Concept & Prompt Alignment (max 2.5): How well the story embraces the prompt and delivers a compelling concept.
2. Narrative Craft & Structure (max 2.5): Cohesion, pacing, clarity, and overall storytelling craft.
3. Character/Imagery & Emotional Impact (max 2.5): Depth of characters or imagery and the emotional journey provided.
4. Style & Language (max 2.5): Command of language, voice, and stylistic polish.
Document the reasoning behind each component score, then report a `total_score` out of 10.0 equal to the sum of the awarded points.
### Response Format ###
Respond **only** with a JSON object containing:
* **"total_score"** — the sum of the awarded points (float).
* **"per_item"** — an array of objects matching the rubric above, each with:
  * **"title"** — the rubric item title.
  * **"awarded"** — the points you are awarding (float).
  * **"max"** — the maximum points available for that item (float).
  * **"reason"** — a concise explanation referencing the findings that justify the awarded score.
Ensure that "total_score" equals the sum of the "awarded" values.""",
).strip()


def get_compare_guidelines(dataset_key: str) -> str:
    if dataset_key == "creative_writing":
        return CREATIVE_COMPARE_GUIDELINES
    return IMO_COMPARE_GUIDELINES


def get_score_guidelines(dataset_key: str) -> str:
    if dataset_key == "creative_writing":
        return CREATIVE_SCORE_GUIDELINES
    return IMO_SCORE_GUIDELINES


@dataclass
class RunCandidate:
    model: str
    run_id: str
    score: Optional[float]
    solution: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Problem:
    problem_id: int
    statement: str
    rubric: Sequence[Dict[str, Any]]
    runs: Sequence[RunCandidate]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def format_candidate_solution(run: RunCandidate) -> str:
    """Combine metadata and solution text for prompting."""
    details: List[str] = []

    title = run.metadata.get("title")
    if title:
        details.append(f"Title: {title}")
    author = run.metadata.get("author")
    if author:
        details.append(f"Author: {author}")
    award = run.metadata.get("award")
    if award:
        details.append(f"Award: {award}")
    source = run.metadata.get("source")
    if source:
        details.append(f"Source: {source}")

    link = None
    nested_meta = run.metadata.get("metadata")
    if isinstance(nested_meta, Mapping):
        link = nested_meta.get("story_url") or nested_meta.get("url")
    if link:
        details.append(f"URL: {link}")

    if details:
        return "\n".join(details + ["", run.solution])
    return run.solution


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
        score_a = run_a.score if run_a.score is not None else float("-inf")
        score_b = run_b.score if run_b.score is not None else float("-inf")
        if math.isfinite(score_a) and math.isfinite(score_b) and math.isclose(score_a, score_b):
            return 0
        if score_a == score_b:
            return 0
        return 1 if score_a > score_b else -1


class LLMJudge(Judge):
    """
    Judge that queries an LLM to compare two solutions.

    Requires the OpenAI Python package and a valid `OPENAI_API_KEY`.
    """

    def __init__(
        self,
        model: str,
        *,
        dataset_key: str,
        temperature: float = 0.0,
        max_workers: int = 4,
    ) -> None:
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
        self._dataset_key = dataset_key

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def compare(self, problem: Problem, run_a: RunCandidate, run_b: RunCandidate) -> int:
        guidelines = get_compare_guidelines(self._dataset_key)
        rubric_section = ""
        if problem.rubric:
            rubric_lines = []
            for idx, item in enumerate(problem.rubric, start=1):
                title = item.get("title") or ""
                desc = item.get("description") or ""
                max_points = item.get("max_points")
                max_info = f" (max {max_points})" if max_points is not None else ""
                rubric_lines.append(f"{idx}. {title}{max_info}: {desc}")
            rubric_body = "\n".join(rubric_lines)
            rubric_section = f"### Rubric ###\n{rubric_body}\n======================================================================\n"
        if self._dataset_key == "creative_writing":
            task_role = "a lead judge for a creative writing competition"
            workflow_noun = "evaluation"
            decision_focus = "creative quality, prompt alignment, and execution"
        else:
            task_role = "an IMO grader"
            workflow_noun = "verification"
            decision_focus = "rigor, correctness, and completeness"

        prompt = textwrap.dedent(
            f"""\
{guidelines}

======================================================================
### Problem ###
{problem.statement}
======================================================================
{rubric_section}### Solution A ###
{format_candidate_solution(run_a)}
======================================================================
### Solution B ###
{format_candidate_solution(run_b)}
======================================================================
### Verification Task Reminder ###
Your task is to act as {task_role}. Independently perform the full {workflow_noun} workflow described above for **each** solution. Internally prepare the Summary and Detailed Log for Solution A and Solution B before making any comparison. Base your final decision solely on that analysis.
For this interface, do NOT output the summaries or logs. Instead, decide which solution is superior in terms of {decision_focus}.
Respond with a single character only:
- A : Solution A is better.
- B : Solution B is better.
- T : The solutions are tied in quality.
"""
        ).strip()
        response = self._client.responses.create(  # type: ignore
            model=self._model,
            input=prompt,
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

    def __init__(
        self,
        model: str,
        *,
        dataset_key: str,
        temperature: float = 0.0,
        max_workers: int = 4,
    ) -> None:
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
        self._dataset_key = dataset_key

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def score(self, problem: Problem, run: RunCandidate) -> float:
        guidelines = get_score_guidelines(self._dataset_key)
        if problem.rubric:
            effective_rubric = list(problem.rubric)
        elif self._dataset_key == "creative_writing":
            effective_rubric = [
                {
                    "title": "Concept & Prompt Alignment",
                    "description": "How well the story embraces the prompt and delivers a compelling concept.",
                    "max_points": 2.5,
                },
                {
                    "title": "Narrative Craft & Structure",
                    "description": "Cohesion, pacing, clarity, and overall storytelling craft.",
                    "max_points": 2.5,
                },
                {
                    "title": "Character/Imagery & Emotional Impact",
                    "description": "Depth of characters or imagery and the emotional journey provided.",
                    "max_points": 2.5,
                },
                {
                    "title": "Style & Language",
                    "description": "Voice, word choice, and stylistic polish.",
                    "max_points": 2.5,
                },
            ]
        else:
            effective_rubric = [
                {
                    "title": "Overall Quality",
                    "description": "Assess the overall correctness, completeness, and rigor of the solution.",
                    "max_points": 10.0,
                }
            ]
        rubric_lines = []
        total_points = 0.0
        for item in effective_rubric:
            title = item.get("title") or ""
            desc = item.get("description") or ""
            max_points = item.get("max_points")
            if max_points is None:
                max_points = 1.0
            total_points += max_points
            rubric_lines.append(f"- {title} (max {max_points}): {desc}")
        rubric_section = ""
        if rubric_lines:
            rubric_body = "\n".join(rubric_lines)
            rubric_section = f"### Rubric ###\n{rubric_body}\n======================================================================\n"

        prompt = textwrap.dedent(
            f"""\
{guidelines}

======================================================================
### Problem ###
{problem.statement}
======================================================================
### Maximum Score ###
The maximum possible score is {total_points}. Partial credit is allowed when justified by the solution.
======================================================================
{rubric_section}### Solution ###
{format_candidate_solution(run)}
======================================================================
### Scoring Task Reminder ###
Your task is to act as {"a lead judge for a creative writing competition" if self._dataset_key == "creative_writing" else "an IMO grader"}. Perform the full {"evaluation" if self._dataset_key == "creative_writing" else "verification"} workflow described above, then translate your findings into the rubric scores exactly as specified in the Response Format.
"""
        ).strip()
        response = self._client.responses.create(  # type: ignore
            model=self._model,
            input=prompt,
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
        #print(content_text, 42)
        json_text = content_text
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
                raw_score = payload.get("score")
                score: Optional[float]
                try:
                    score = float(raw_score) if raw_score is not None else None
                except (TypeError, ValueError):
                    score = None
                solution = payload.get("solve", "") or ""
                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"solve", "score"}
                }
                candidates.append(
                    RunCandidate(
                        model=model_name,
                        run_id=run_id,
                        score=score,
                        solution=solution,
                        metadata=metadata,
                    )
                )
        problems.append(
            Problem(
                problem_id=int(problem_entry["problem"]),
                statement=problem_entry["statement"],
                rubric=problem_entry.get("rubric", []),
                runs=candidates,
                metadata=problem_entry.get("metadata", {}),
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
        score_display = f"{run.score:>7.2f}" if run.score is not None else f"{'N/A':>7}"
        lines.append(
            f"{rank:<4} {run.model:<25} {run.run_id:<6} {wins[idx]:>6.1f} {score_display}"
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
        score_display = f"{run.score:>7.2f}" if run.score is not None else f"{'N/A':>7}"
        lines.append(
            f"{rank:<4} {run.model:<25} {run.run_id:<6} {ratings[idx]:>8.1f} {score_display}"
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
    parser = argparse.ArgumentParser(
        description="Rank candidate solutions using different schemes (supports math and creative writing datasets)."
    )
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
    dataset_override: Optional[Path] = args.solutions_json

    problems = load_dataset(args.dataset, dataset_override)

    judge: Judge | None = None
    scorer: LLMRubricScorer | None = None
    if args.ranking_scheme in {"pairwise", "elo"}:
        if args.judge == "llm":
            judge = LLMJudge(
                args.llm_model,
                dataset_key=args.dataset,
                max_workers=args.llm_workers,
            )
        else:
            judge = ScoreJudge()
    if args.ranking_scheme == "llm_score":
        scorer = LLMRubricScorer(
            args.llm_model,
            dataset_key=args.dataset,
            max_workers=args.llm_workers,
        )

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
            if run.metadata:
                record["metadata"] = run.metadata
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

        scores_all = [run.score if run.score is not None else 0.0 for run in problem.runs]
        total_score = sum(scores_all)
        max_score = max(scores_all, default=0.0)
        rank_sum_actual = sum(rec["rank"] for rec in ranking_records)
        rank_max = ranking_records[-1]["rank"] if ranking_records else 0
        num_runs = len(problem.runs)

        indices = list(range(num_runs))
        score_values = {idx: scores_all[idx] for idx in indices}
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
                "metadata": problem.metadata,
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
