from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from creative_dataset import (
    CreativeDatasetError,
    convert_prompts_to_ranking_document,
    scrape_reedsy_prompts,
    write_creative_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Reedsy creative-writing contests into a ranking dataset JSON."
    )
    parser.add_argument(
        "contest_ids",
        metavar="CONTEST_ID",
        type=int,
        nargs="+",
        help="Reedsy contest IDs to scrape (e.g. 316 315).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reedsy_dataset.json"),
        help="Destination JSON path (default: %(default)s).",
    )
    parser.add_argument(
        "--regular-count",
        type=int,
        default=2,
        metavar="N",
        help="Number of non-awarded stories to include per contest (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prompts = scrape_reedsy_prompts(
            args.contest_ids,
            regular_story_count=args.regular_count,
        )
        document = convert_prompts_to_ranking_document(
            prompts,
            dataset_key="reedsy_contests",
            dataset_name="Reedsy Writing Contests",
        )
        write_creative_dataset(document, args.output)
    except CreativeDatasetError as error:
        print(f"Error: {error}")
        return 1

    prompt_count = len(document.get("problems", []))
    print(
        f"Wrote {prompt_count} contest prompt{'s' if prompt_count != 1 else ''} "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
