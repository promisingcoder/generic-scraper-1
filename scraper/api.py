"""Public API for the scraper library.

This module provides the primary interface that users interact with:

- **Scraper**: The main class, used as an async context manager.
- **define_fields()**: Convenience helper for creating FieldDefinition lists.
- **scrape_sync()**: Synchronous wrapper for simple single-URL scraping.

Usage::

    from scraper import Scraper, define_fields

    fields = define_fields(
        ("company_name", "The name of the company", "string"),
        ("emails", "All contact email addresses", "list[string]"),
    )

    async with Scraper(backend="http") as s:
        # Single URL
        result = await s.scrape("https://example.com", fields)
        print(result.data)

        # Multiple URLs
        async for result in s.scrape_many(["https://a.com", "https://b.com"], fields):
            print(result.data)

        # Crawl mode
        async for result in s.crawl("https://example.com", fields, max_pages=20):
            print(result.data)
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, AsyncGenerator, List, Optional, Sequence, Tuple, Union

from scraper.browser_backend import BrowserBackend
from scraper.config import load_config
from scraper.exceptions import ConfigError
from scraper.http_backend import HTTPBackend
from scraper.models import FieldDefinition, ScraperConfig, ScrapeResult

__all__ = [
    "Scraper",
    "define_fields",
    "scrape_sync",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def define_fields(
    *field_tuples: Union[
        Tuple[str, str],
        Tuple[str, str, str],
    ],
) -> List[FieldDefinition]:
    """Create a list of FieldDefinition objects from tuples.

    Each tuple can be:
    - ``(name, description)`` -- defaults to type ``"string"``
    - ``(name, description, type)`` -- explicit type

    Args:
        *field_tuples: Variable number of 2- or 3-tuples.

    Returns:
        A list of FieldDefinition objects.

    Raises:
        ConfigError: If any tuple has invalid length or invalid field type.

    Examples:
        >>> fields = define_fields(
        ...     ("company_name", "The company's official name"),
        ...     ("phone", "Main phone number", "string"),
        ...     ("employees", "List of employee names", "list[string]"),
        ... )
        >>> len(fields)
        3
        >>> fields[0].name
        'company_name'
        >>> fields[0].type
        'string'
    """
    result: List[FieldDefinition] = []

    for i, t in enumerate(field_tuples):
        if len(t) == 2:
            name, description = t
            field_type = "string"
        elif len(t) == 3:
            name, description, field_type = t
        else:
            raise ConfigError(
                f"Field tuple at index {i} has {len(t)} elements; expected 2 or 3. "
                f"Format: (name, description) or (name, description, type)"
            )

        result.append(FieldDefinition(name=name, description=description, type=field_type))

    return result


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------


class Scraper:
    """Main scraper class providing single-URL, multi-URL, and crawl modes.

    Must be used as an async context manager to ensure proper resource
    cleanup (HTTP sessions, browser processes).

    Args:
        backend: Fetching backend -- ``"http"`` (default) for static pages,
                 ``"browser"`` for JS-rendered pages.
        **config_kwargs: Additional configuration passed to ``ScraperConfig``.
                         See ``ScraperConfig`` for all available options.

    Examples:
        >>> async with Scraper(backend="http", model="openai/gpt-4o-mini") as s:
        ...     result = await s.scrape("https://example.com", fields)
        ...     print(result.data)

        >>> async with Scraper(backend="browser", headless=True) as s:
        ...     async for result in s.crawl("https://example.com", fields):
        ...         print(result.data)
    """

    def __init__(self, backend: str = "http", **config_kwargs: Any) -> None:
        """Initialize the scraper with the given backend and config.

        Args:
            backend: ``"http"`` or ``"browser"``.
            **config_kwargs: Passed to ScraperConfig (e.g., model, api_keys,
                             max_concurrent, timeout, headless, stealth, etc.).

        Raises:
            ConfigError: If backend is not ``"http"`` or ``"browser"``.
        """
        if backend not in ("http", "browser"):
            raise ConfigError(
                f"Invalid backend '{backend}'. Must be 'http' or 'browser'."
            )

        config_kwargs["backend"] = backend
        self._config = load_config(**config_kwargs)
        self._backend: Optional[Union[HTTPBackend, BrowserBackend]] = None
        self._entered = False

    @property
    def config(self) -> ScraperConfig:
        """Return the current scraper configuration (read-only)."""
        return self._config

    async def __aenter__(self) -> Scraper:
        """Start the scraper backend.

        Creates and starts the appropriate backend (HTTP or Browser)
        based on the configured backend type.

        Returns:
            self
        """
        if self._config.backend == "browser":
            self._backend = BrowserBackend(self._config)
        else:
            self._backend = HTTPBackend(self._config)

        await self._backend.__aenter__()
        self._entered = True
        logger.info("Scraper started with %s backend", self._config.backend)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shut down the scraper backend.

        Ensures all resources (HTTP sessions, browser processes) are
        properly cleaned up, even if an error occurred.
        """
        if self._backend is not None:
            await self._backend.__aexit__(exc_type, exc_val, exc_tb)
            self._backend = None
        self._entered = False
        logger.info("Scraper shut down")

    def _ensure_entered(self) -> Union[HTTPBackend, BrowserBackend]:
        """Ensure the context manager has been entered.

        Returns:
            The active backend instance.

        Raises:
            RuntimeError: If the scraper is used outside a context manager.
        """
        if not self._entered or self._backend is None:
            raise RuntimeError(
                "Scraper must be used as an async context manager: "
                "'async with Scraper() as s: ...'"
            )
        return self._backend

    # -------------------------------------------------------------------
    # Single-URL mode
    # -------------------------------------------------------------------

    async def scrape(
        self,
        url: str,
        fields: List[FieldDefinition],
    ) -> ScrapeResult:
        """Scrape a single URL and extract the requested fields.

        This is the simplest mode. Fetches one page, runs LLM extraction,
        and returns the result.

        Args:
            url: The URL to scrape.
            fields: List of FieldDefinition objects describing what to extract.

        Returns:
            A ScrapeResult with the extracted data.

        Raises:
            RuntimeError: If the scraper is not in a context manager.
        """
        backend = self._ensure_entered()
        return await backend.scrape_one(url, fields)

    # -------------------------------------------------------------------
    # Multi-URL mode
    # -------------------------------------------------------------------

    async def scrape_many(
        self,
        urls: List[str],
        fields: List[FieldDefinition],
    ) -> AsyncGenerator[ScrapeResult, None]:
        """Scrape multiple URLs, yielding results as each completes.

        URLs are processed concurrently (up to ``max_concurrent``).
        Results are yielded as they finish -- the caller does not wait
        for all URLs to complete before receiving the first result.

        Duplicate URLs are silently skipped.

        Args:
            urls: List of URL strings to scrape.
            fields: List of FieldDefinition objects describing what to extract.

        Yields:
            ScrapeResult for each URL as extraction completes.

        Raises:
            RuntimeError: If the scraper is not in a context manager.
        """
        backend = self._ensure_entered()
        async for result in backend.scrape_many(urls, fields):
            yield result

    # -------------------------------------------------------------------
    # Crawl mode
    # -------------------------------------------------------------------

    async def crawl(
        self,
        start_url: str,
        fields: List[FieldDefinition],
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> AsyncGenerator[ScrapeResult, None]:
        """Crawl starting from a URL, following links and extracting from each page.

        The crawler follows internal links within the same domain,
        prioritizing high-value paths (about, contact, team, etc.).
        Results are yielded incrementally as pages are crawled.

        Completion detection:
        - If ALL fields are scalar and all have been found with non-null
          values, crawling stops early (all data found).
        - If ANY field is a list type, crawling continues until
          ``max_pages`` is reached or no more links remain.

        Args:
            start_url: The URL to start crawling from.
            fields: List of FieldDefinition objects describing what to extract.
            max_pages: Maximum pages to crawl (overrides config.max_pages).
            max_depth: Maximum crawl depth (overrides config.max_depth).

        Yields:
            ScrapeResult for each crawled page as extraction completes.

        Raises:
            RuntimeError: If the scraper is not in a context manager.
        """
        backend = self._ensure_entered()
        async for result in backend.crawl(
            start_url,
            fields,
            max_pages=max_pages,
            max_depth=max_depth,
        ):
            yield result

    # -------------------------------------------------------------------
    # Sync wrapper
    # -------------------------------------------------------------------

    def scrape_sync(
        self,
        url: str,
        fields: List[FieldDefinition],
        backend: str = "http",
        **config_kwargs: Any,
    ) -> ScrapeResult:
        """Synchronous wrapper for single-URL scraping.

        This is a convenience method for simple scripts that do not need
        async. It creates its own event loop, runs the scrape, and returns.

        NOTE: This method creates a NEW backend and shuts it down after
        the single scrape. For multiple scrapes, use the async API with
        a context manager for better performance.

        Args:
            url: The URL to scrape.
            fields: List of FieldDefinition objects.
            backend: Backend override (defaults to config or "http").
            **config_kwargs: Additional config overrides.

        Returns:
            A ScrapeResult with extracted data.
        """

        async def _run() -> ScrapeResult:
            async with Scraper(backend=backend, **config_kwargs) as s:
                return await s.scrape(url, fields)

        return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Module-level sync helper
# ---------------------------------------------------------------------------


def scrape_sync(
    url: str,
    fields: List[FieldDefinition],
    backend: str = "http",
    **config_kwargs: Any,
) -> ScrapeResult:
    """Synchronous convenience function for single-URL scraping.

    Creates a Scraper, scrapes one URL, and returns the result.
    For multiple URLs or crawling, use the async ``Scraper`` class.

    Args:
        url: The URL to scrape.
        fields: List of FieldDefinition objects.
        backend: ``"http"`` (default) or ``"browser"``.
        **config_kwargs: Passed to ScraperConfig.

    Returns:
        A ScrapeResult with extracted data.

    Examples:
        >>> from scraper import scrape_sync, define_fields
        >>> fields = define_fields(("title", "Page title"))
        >>> result = scrape_sync("https://example.com", fields)
        >>> print(result.data)
    """

    async def _run() -> ScrapeResult:
        async with Scraper(backend=backend, **config_kwargs) as s:
            return await s.scrape(url, fields)

    return asyncio.run(_run())
