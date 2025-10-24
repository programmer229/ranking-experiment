from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from reedsy_prompts_scraper import (
    ScraperError as ReedsyScraperError,
    scrape_contests,
)

from literary_hub_scraper import (
    ScraperError as LiteraryHubScraperError,
    scrape_stories_from_pages,
)


LOGGER = logging.getLogger(__name__)


class CreativeDatasetError(RuntimeError):
    """Raised when creative dataset construction fails."""


@dataclass
class CreativeStory:
    source: str
    story_id: str
    title: str
    author: Optional[str]
    text: str
    score: Optional[float] = None
    award: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreativePrompt:
    prompt_id: str
    title: str
    prompt: str
    source: str
    stories: List[CreativeStory]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ensure_text(value: Optional[str]) -> bool:
    return bool(value and value.strip())


def scrape_reedsy_prompts(
    contest_ids: Sequence[int],
    *,
    regular_story_count: int = 2,
) -> List[CreativePrompt]:
    try:
        contests = scrape_contests(contest_ids, regular_story_count=regular_story_count)
    except ReedsyScraperError as err:
        raise CreativeDatasetError(f"Failed to scrape Reedsy contests: {err}") from err

    prompts: List[CreativePrompt] = []
    for contest in contests:
        stories: List[CreativeStory] = []
        for entry in contest.get("stories", []):
            text = entry.get("text") or ""
            if not _ensure_text(text):
                continue
            story_id = entry.get("slug") or entry.get("story_url") or entry.get("title")
            if not story_id:
                continue
            stories.append(
                CreativeStory(
                    source="reedsy",
                    story_id=str(story_id),
                    title=entry.get("title") or str(story_id),
                    author=entry.get("author"),
                    text=text,
                    score=float(entry.get("points", 0) or 0.0),
                    award=entry.get("award"),
                    metadata={
                        "story_url": entry.get("story_url"),
                        "author_url": entry.get("author_url"),
                        "award_text": entry.get("award_text"),
                        "blurb": entry.get("blurb"),
                    },
                )
            )
        if not stories:
            continue

        prompt_text = contest.get("prompt") or ""
        if not _ensure_text(prompt_text):
            prompt_text = contest.get("title") or "Creative Writing Prompt"

        prompts.append(
            CreativePrompt(
                prompt_id=str(contest.get("contest_id") or contest.get("title") or len(prompts) + 1),
                title=contest.get("title") or f"Reedsy Contest {contest.get('contest_id')}",
                prompt=prompt_text,
                source="reedsy",
                stories=stories,
                metadata={
                    "contest_id": contest.get("contest_id"),
                    "contest_url": contest.get("contest_url"),
                },
            )
        )
    return prompts


def scrape_literary_hub_prompts(
    base_path: str,
    page_numbers: Sequence[int],
    *,
    delay_seconds: float = 1.0,
) -> List[CreativePrompt]:
    try:
        stories_raw = scrape_stories_from_pages(
            base_path,
            page_numbers,
            delay_seconds=delay_seconds,
        )
    except LiteraryHubScraperError as err:
        raise CreativeDatasetError(f"Failed to scrape Literary Hub stories: {err}") from err

    if not stories_raw:
        return []

    stories: List[CreativeStory] = []
    for story in stories_raw:
        if not _ensure_text(story.text):
            continue
        stories.append(
            CreativeStory(
                source="literary_hub",
                story_id=story.slug,
                title=story.title,
                author=story.author,
                text=story.text,
                score=None,
                award=None,
                metadata={"url": story.url},
            )
        )

    if not stories:
        return []

    prompt = CreativePrompt(
        prompt_id="literary_hub_short_stories",
        title="Literary Hub Short Stories",
        prompt=(
            "These stories were curated from the Literary Hub short story archive. "
            "Evaluate the creative quality, narrative craft, and overall impact of each piece."
        ),
        source="literary_hub",
        stories=stories,
        metadata={"base_path": base_path, "pages": list(page_numbers)},
    )
    return [prompt]


def convert_prompts_to_ranking_document(
    prompts: Sequence[CreativePrompt],
    *,
    dataset_key: str = "creative_writing",
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not prompts:
        raise CreativeDatasetError("No prompts supplied to convert into a ranking document.")

    problems: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        models: Dict[str, Dict[str, Any]] = {}
        for story_index, story in enumerate(prompt.stories, start=1):
            if not _ensure_text(story.text):
                continue
            model_key = story.author or f"{story.source}:{story.story_id}"
            run_id = f"story{story_index}"
            run_payload: Dict[str, Any] = {
                "solve": story.text,
                "score": story.score,
                "title": story.title,
                "author": story.author,
                "award": story.award,
                "source": story.source,
                "story_id": story.story_id,
                "metadata": story.metadata,
            }
            models.setdefault(model_key, {})
            models[model_key][run_id] = run_payload

        if not models:
            continue

        statement_parts = [
            f"Source: {prompt.source}",
            f"Prompt Title: {prompt.title}",
        ]
        if prompt.prompt:
            statement_parts.append("")
            statement_parts.append(prompt.prompt)
        statement = "\n".join(statement_parts)

        problem_identifier: Any = prompt.prompt_id
        try:
            problem_identifier = int(str(prompt.prompt_id))
        except ValueError:
            problem_identifier = index

        problems.append(
            {
                "problem": problem_identifier,
                "statement": statement,
                "rubric": [],
                "models": models,
                "metadata": prompt.metadata,
            }
        )

    if not problems:
        raise CreativeDatasetError("No problems constructed from creative prompts.")

    return {
        "competition": dataset_key,
        "nice_name": dataset_name or "Creative Writing Evaluation",
        "problems": problems,
    }


def build_creative_dataset(
    *,
    reedsy_contests: Sequence[int] | None = None,
    reedsy_regular_story_count: int = 2,
    literary_hub_base_path: Optional[str] = None,
    literary_hub_pages: Sequence[int] | None = None,
) -> Dict[str, Any]:
    prompts: List[CreativePrompt] = []

    if reedsy_contests:
        prompts.extend(
            scrape_reedsy_prompts(
                reedsy_contests,
                regular_story_count=reedsy_regular_story_count,
            )
        )

    if literary_hub_base_path and literary_hub_pages:
        prompts.extend(
            scrape_literary_hub_prompts(
                literary_hub_base_path,
                literary_hub_pages,
            )
        )

    if not prompts:
        raise CreativeDatasetError("No creative prompts were gathered from the specified sources.")

    return convert_prompts_to_ranking_document(prompts)


def write_creative_dataset(document: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2))
    LOGGER.info("Wrote creative dataset with %s prompts to %s", len(document.get("problems", [])), destination)


__all__ = [
    "CreativeDatasetError",
    "CreativePrompt",
    "CreativeStory",
    "build_creative_dataset",
    "convert_prompts_to_ranking_document",
    "scrape_literary_hub_prompts",
    "scrape_reedsy_prompts",
    "write_creative_dataset",
]
