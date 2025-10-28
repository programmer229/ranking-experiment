from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, List, Optional, Sequence
from time import sleep

import requests
from bs4 import BeautifulSoup, Tag

from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://blog.reedsy.com"
CONTEST_PATH_TEMPLATE = "/creative-writing-prompts/contests/{contest_id}/"
STORY_PATH_PREFIX = "/short-story/"


class ScraperError(RuntimeError):
    """Raised when the Reedsy scraper cannot extract the expected content."""


@dataclass
class StoryRecord:
    slug: str
    title: str
    author: str
    author_url: str
    story_url: str
    award: str
    award_text: str
    points: int
    blurb: Optional[str] = None
    text: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_url(path: str) -> str:
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{BASE_URL}{path}"


def _parse_retry_after(header_value: Optional[str], default_wait: float) -> float:
    if not header_value:
        return default_wait
    try:
        return float(header_value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(header_value)
        except (TypeError, ValueError):
            return default_wait
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delta = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(delta, default_wait)


def _fetch_html(url: str, *, max_attempts: int = 8) -> str:
    LOGGER.debug("Fetching %s", url)
    backoff = 2.0
    last_error: Optional[requests.HTTPError] = None

    for attempt in range(1, max_attempts + 1):
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0.0.0 Safari/537.36"
                )
            },
            timeout=30,
        )

        if response.status_code == 429 and attempt < max_attempts:
            wait_seconds = _parse_retry_after(
                response.headers.get("Retry-After"), default_wait=backoff
            )
            LOGGER.info(
                "429 Too Many Requests for %s. Sleeping %.1fs before retry (%s/%s).",
                url,
                wait_seconds,
                attempt,
                max_attempts,
            )
            sleep(wait_seconds)
            backoff = min(backoff * 2, 60.0)
            continue

        try:
            response.raise_for_status()
            return response.text
        except requests.HTTPError as err:
            last_error = err
            LOGGER.debug("Attempt %s for %s failed: %s", attempt, url, err)
            if attempt == max_attempts:
                break
            sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    if last_error is None:
        raise ScraperError(f"Failed to fetch {url} after {max_attempts} attempts.")
    raise last_error


def _normalise_spaces(value: str) -> str:
    return " ".join(value.split())


def _extract_prompt(main: Tag) -> str:
    article = main.select_one(".article")
    if not article:
        raise ScraperError("Could not locate the contest prompt copy.")

    paragraphs: List[str] = []
    for node in article.find_all("p"):
        text = node.get_text(strip=True)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _classify_award(raw_text: str) -> tuple[str, int]:
    if not raw_text:
        return "regular", 0
    text = raw_text.lower()
    if "winner" in text:
        return "winner", 10
    shortlist_tokens = ("shortlist", "short-listed", "short-listed", "runner", "featured")
    if any(token in text for token in shortlist_tokens):
        return "featured", 5
    return "regular", 0


def _parse_story_submission(submission: Tag) -> StoryRecord:
    header_link = submission.select_one("h4 a")
    if not header_link:
        raise ScraperError("Submission block missing the story title link.")
    title = _normalise_spaces(header_link.get_text(strip=True))
    story_path = header_link.get("href") or ""
    story_url = _build_url(story_path)

    slug = story_url.rstrip("/").split("/")[-1]

    author_link = submission.select_one("a[href*='/creative-writing-prompts/author/']")
    if not author_link:
        raise ScraperError(f"Story '{title}' is missing author information.")
    author = _normalise_spaces(author_link.get_text(strip=True))
    author_url = _build_url(author_link.get("href") or "")

    award_element = submission.select_one("p.fgColor-warning, p.fgColor-muted")
    award_text = award_element.get_text(strip=True) if award_element else ""
    award, points = _classify_award(award_text)

    blurb_element = submission.select_one(".text-prompt")
    blurb = None
    if blurb_element:
        blurb = _normalise_spaces(blurb_element.get_text(strip=True))

    return StoryRecord(
        slug=slug,
        title=title,
        author=author,
        author_url=author_url,
        story_url=story_url,
        award=award,
        award_text=award_text,
        points=points,
        blurb=blurb,
    )


def _extract_story_text(story_url: str) -> str:
    html = _fetch_html(story_url)
    soup = BeautifulSoup(html, "html.parser")

    article = soup.select_one("article.article.story-typeset")
    if not article:
        article = soup.select_one("article.article")
    if not article:
        raise ScraperError(f"Could not locate story body for {story_url}")

    paragraphs: List[str] = []
    for paragraph in article.find_all("p"):
        text = paragraph.get_text()
        text = text.strip()
        if not text:
            continue
        paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _pick_regular_stories(
    stories: Sequence[StoryRecord],
    skip_slugs: Iterable[str],
    desired_count: int,
) -> List[StoryRecord]:
    skip_set = set(skip_slugs)
    regulars: List[StoryRecord] = []
    for story in stories:
        if story.slug in skip_set:
            continue
        if story.award != "regular":
            continue
        regulars.append(story)
        if len(regulars) >= desired_count:
            break
    return regulars


def scrape_contest(contest_id: int, regular_story_count: int = 2) -> dict:
    contest_url = _build_url(CONTEST_PATH_TEMPLATE.format(contest_id=contest_id))
    html = _fetch_html(contest_url)
    soup = BeautifulSoup(html, "html.parser")

    main = soup.select_one("main")
    if not main:
        raise ScraperError(f"Contest page {contest_id} is missing the <main> section.")

    title_element = main.select_one("h1")
    if not title_element:
        raise ScraperError(f"Contest page {contest_id} lacks a title heading.")
    contest_title = _normalise_spaces(title_element.get_text(strip=True))

    prompt_text = _extract_prompt(main)

    submissions_container = soup.select_one("#submissions-container")
    if not submissions_container:
        raise ScraperError(f"Contest page {contest_id} has no submissions container.")

    submission_elements = submissions_container.select(".submission")
    if not submission_elements:
        raise ScraperError(f"Contest page {contest_id} has no submission entries.")

    stories = [_parse_story_submission(sub) for sub in submission_elements]

    selected: List[StoryRecord] = []
    winner = next((story for story in stories if story.award == "winner"), None)
    if winner:
        selected.append(winner)

    featured = next((story for story in stories if story.award == "featured"), None)
    if featured and featured.slug != (winner.slug if winner else None):
        selected.append(featured)

    regulars = _pick_regular_stories(
        stories,
        skip_slugs=[story.slug for story in selected],
        desired_count=regular_story_count,
    )
    selected.extend(regulars)

    for story in selected:
        try:
            story.text = _extract_story_text(story.story_url)
            sleep(1.0)
        except requests.HTTPError as err:
            raise ScraperError(
                f"Failed to fetch story '{story.title}' ({story.story_url}): {err}"
            ) from err

    return {
        "contest_id": contest_id,
        "contest_url": contest_url,
        "title": contest_title,
        "prompt": prompt_text,
        "stories": [story.to_dict() for story in selected],
    }


def scrape_contests(contest_ids: Sequence[int], regular_story_count: int = 2) -> List[dict]:
    results: List[dict] = []
    for contest_id in tqdm(contest_ids, desc="Reedsy contests", unit="contest"):
        results.append(scrape_contest(contest_id, regular_story_count=regular_story_count))
    return results


def write_contests_to_json(
    contest_ids: Sequence[int],
    destination_path: str,
    regular_story_count: int = 2,
) -> None:
    contests = scrape_contests(contest_ids, regular_story_count=regular_story_count)
    with open(destination_path, "w", encoding="utf-8") as handle:
        json.dump(contests, handle, ensure_ascii=False, indent=2)


__all__ = [
    "ScraperError",
    "StoryRecord",
    "scrape_contest",
    "scrape_contests",
    "write_contests_to_json",
]
