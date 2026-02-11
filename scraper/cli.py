"""Command-line interface for the scraper library.

Provides a simple CLI for scraping URLs from the terminal::

    # Single URL
    scraper scrape https://example.com --field "name:Company name:string" --field "emails:Contact emails:list[string]"

    # Multiple URLs
    scraper scrape-many https://a.com https://b.com --field "title:Page title"

    # Crawl mode
    scraper crawl https://example.com --field "name:Company name" --max-pages 20

    # Specify backend
    scraper scrape https://example.com --field "name:Company name" --backend browser

Entry point is configured in pyproject.toml as ``scraper = "scraper.cli:main"``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
from typing import List

from scraper.api import Scraper, define_fields
from scraper.models import FieldDefinition

__all__ = ["main"]


def _parse_field_arg(field_str: str) -> tuple[str, str, str]:
    """Parse a field argument string like 'name:Description:type'.

    Format: ``name:description[:type]``
    - name and description are required
    - type defaults to "string" if omitted

    Args:
        field_str: Colon-separated field specification.

    Returns:
        Tuple of (name, description, type).

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    parts = field_str.split(":", maxsplit=2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"Invalid field format: '{field_str}'. "
            "Expected 'name:description' or 'name:description:type'"
        )

    name = parts[0].strip()
    description = parts[1].strip()
    field_type = parts[2].strip() if len(parts) == 3 else "string"

    if not name:
        raise argparse.ArgumentTypeError("Field name cannot be empty")
    if not description:
        raise argparse.ArgumentTypeError("Field description cannot be empty")

    return (name, description, field_type)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="scraper",
        description="Web scraper with LLM-based extraction",
    )
    parser.add_argument(
        "--backend",
        choices=["http", "browser"],
        default="http",
        help="Fetching backend (default: http)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model in litellm format (default: openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Scraping mode")

    # --- scrape (single URL) ---
    scrape_parser = subparsers.add_parser("scrape", help="Scrape a single URL")
    scrape_parser.add_argument("url", help="URL to scrape")
    scrape_parser.add_argument(
        "--field", "-f",
        action="append",
        required=True,
        dest="fields",
        help="Field to extract: 'name:description[:type]' (repeatable)",
    )

    # --- scrape-many (multiple URLs) ---
    many_parser = subparsers.add_parser("scrape-many", help="Scrape multiple URLs")
    many_parser.add_argument("urls", nargs="+", help="URLs to scrape")
    many_parser.add_argument(
        "--field", "-f",
        action="append",
        required=True,
        dest="fields",
        help="Field to extract: 'name:description[:type]' (repeatable)",
    )
    many_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests (default: 5)",
    )

    # --- crawl ---
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a site")
    crawl_parser.add_argument("url", help="Starting URL")
    crawl_parser.add_argument(
        "--field", "-f",
        action="append",
        required=True,
        dest="fields",
        help="Field to extract: 'name:description[:type]' (repeatable)",
    )
    crawl_parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages to crawl (default: 50)",
    )
    crawl_parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum crawl depth (default: 3)",
    )

    return parser


@contextlib.contextmanager
def _suppress_crawl4ai_stdout():
    """Redirect stdout to devnull during crawl4ai operations.

    crawl4ai 0.8.0 writes progress bars (e.g., '[FETCH]...') directly to
    stdout, polluting JSON output. This context manager suppresses that noise.
    We save a reference to the real stdout so _print_result can write to it.
    """
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield real_stdout
    finally:
        sys.stdout.close()
        sys.stdout = real_stdout


async def _run_scrape(args: argparse.Namespace) -> None:
    """Execute the scrape command."""
    field_tuples = [_parse_field_arg(f) for f in args.fields]
    fields = define_fields(*field_tuples)

    with _suppress_crawl4ai_stdout() as real_stdout:
        async with Scraper(backend=args.backend, model=args.model, verbose=args.verbose) as s:
            result = await s.scrape(args.url, fields)
        _print_result(result, file=real_stdout)


async def _run_scrape_many(args: argparse.Namespace) -> None:
    """Execute the scrape-many command."""
    field_tuples = [_parse_field_arg(f) for f in args.fields]
    fields = define_fields(*field_tuples)

    with _suppress_crawl4ai_stdout() as real_stdout:
        async with Scraper(
            backend=args.backend,
            model=args.model,
            max_concurrent=args.max_concurrent,
            verbose=args.verbose,
        ) as s:
            async for result in s.scrape_many(args.urls, fields):
                _print_result(result, jsonl=True, file=real_stdout)


async def _run_crawl(args: argparse.Namespace) -> None:
    """Execute the crawl command."""
    field_tuples = [_parse_field_arg(f) for f in args.fields]
    fields = define_fields(*field_tuples)

    with _suppress_crawl4ai_stdout() as real_stdout:
        async with Scraper(
            backend=args.backend,
            model=args.model,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            verbose=args.verbose,
        ) as s:
            async for result in s.crawl(args.url, fields, max_pages=args.max_pages, max_depth=args.max_depth):
                _print_result(result, jsonl=True, file=real_stdout)


def _print_result(
    result: "ScrapeResult",
    *,
    jsonl: bool = False,
    file: "io.TextIOBase | None" = None,
) -> None:
    """Print a ScrapeResult as JSON.

    Args:
        result: The ScrapeResult to print.
        jsonl: If True, output compact single-line JSON (JSONL format).
               If False, output pretty-printed JSON.
        file: Output stream (defaults to sys.stdout).
    """
    out = file or sys.stdout
    output = {
        "url": result.url,
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "timestamp": result.timestamp,
    }
    if jsonl:
        print(json.dumps(output, default=str), file=out, flush=True)
    else:
        print(json.dumps(output, indent=2, default=str), file=out)


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "scrape": _run_scrape,
        "scrape-many": _run_scrape_many,
        "crawl": _run_crawl,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
