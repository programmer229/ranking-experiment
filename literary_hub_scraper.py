from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import requests
from bs4 import BeautifulSoup, Tag


LOGGER = logging.getLogger(__name__)

BASE_URL = "https://lithub.com"


class ScraperError(RuntimeError):
    """Raised when the Literary Hub scraper cannot extract the expected content."""


def _normalise_whitespace(value: str) -> str:
    return " ".join(value.split())


def _build_page_url(base_path: str, page_number: int) -> str:
    path = base_path.rstrip("/") + f"{page_number}/"
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{BASE_URL}{path}"


def _fetch_html(url: str, *, session: Optional[requests.Session] = None, timeout: float = 30.0) -> str:
    LOGGER.debug("Fetching %s", url)
    http = session or requests
    response = http.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            )
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def _extract_listing_articles(soup: BeautifulSoup) -> List[Tag]:
    candidates = soup.select("article")
    if candidates:
        return candidates
    return soup.select(".post, .article, .archive-list-item")


@dataclass
class LiteraryHubListingEntry:
    title: str
    url: str
    author: Optional[str]


def _parse_listing_entry(article: Tag) -> Optional[LiteraryHubListingEntry]:
    link = article.find("a", href=True)
    if not link:
        return None
    url = link["href"]
    if not url.startswith("http"):
        url = f"{BASE_URL}{url}"
    title = _normalise_whitespace(link.get_text(strip=True))
    if not title:
        return None
    author: Optional[str] = None
    author_node = article.select_one(".byline, .author, .item-meta .meta-author, .meta-author")
    if author_node:
        author = _normalise_whitespace(author_node.get_text(strip=True))
        author = author.removeprefix("By ").removeprefix("by ").strip()
    return LiteraryHubListingEntry(title=title, url=url, author=author or None)


def _extract_paragraphs(article: Tag) -> List[str]:
    paragraphs: List[str] = []
    for paragraph in article.find_all("p"):
        text = paragraph.get_text()
        text = text.strip()
        if not text:
            continue
        # Skip boilerplate sections that frequently appear at the end of articles.
        lower = text.lower()
        if lower.startswith("via ") or "read next" in lower or "newsletter" in lower:
            break
        paragraphs.append(text)
    return paragraphs


def _parse_story_page(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article")
    if not article:
        raise ScraperError("Unable to locate <article> content on the story page.")
    paragraphs = _extract_paragraphs(article)
    if not paragraphs:
        raise ScraperError("Story page does not contain any paragraphs.")
    return "\n\n".join(paragraphs)


@dataclass
class LiteraryHubStory:
    slug: str
    title: str
    url: str
    author: Optional[str]
    text: str


def scrape_listing_page(
    base_path: str,
    page_number: int,
    *,
    session: Optional[requests.Session] = None,
) -> List[LiteraryHubListingEntry]:
    listing_url = _build_page_url(base_path, page_number)
    html = _fetch_html(listing_url, session=session)
    soup = BeautifulSoup(html, "html.parser")
    articles = _extract_listing_articles(soup)

    entries: List[LiteraryHubListingEntry] = []
    for article in articles:
        entry = _parse_listing_entry(article)
        if not entry:
            continue
        entries.append(entry)
    return entries


def scrape_story(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> LiteraryHubStory:
    html = _fetch_html(url, session=session)
    text = _parse_story_page(html)
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return LiteraryHubStory(
        slug=slug,
        title=title or slug.replace("-", " ").title(),
        url=url,
        author=author,
        text=text,
    )


def scrape_stories_from_pages(
    base_path: str,
    page_numbers: Sequence[int],
    *,
    delay_seconds: float = 1.0,
) -> List[LiteraryHubStory]:
    stories: List[LiteraryHubStory] = []
    seen_urls: set[str] = set()

    with requests.Session() as session:
        for page_number in page_numbers:
            try:
                entries = scrape_listing_page(base_path, page_number, session=session)
            except requests.HTTPError as err:
                LOGGER.warning("Failed to fetch listing page %s: %s", page_number, err)
                continue

            for entry in entries:
                if entry.url in seen_urls:
                    continue
                seen_urls.add(entry.url)
                try:
                    story = scrape_story(
                        entry.url,
                        session=session,
                        title=entry.title,
                        author=entry.author,
                    )
                except (requests.HTTPError, ScraperError) as err:
                    LOGGER.warning("Failed to scrape story %s: %s", entry.url, err)
                    continue

                stories.append(story)
                if delay_seconds:
                    time.sleep(delay_seconds)
    return stories


__all__ = [
    "ScraperError",
    "LiteraryHubListingEntry",
    "LiteraryHubStory",
    "scrape_listing_page",
    "scrape_story",
    "scrape_stories_from_pages",
]
