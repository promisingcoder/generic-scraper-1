"""Browser backend using crawl4ai's AsyncWebCrawler with Playwright.

This backend launches a real Chromium browser via Playwright for fetching
JavaScript-rendered pages. It is heavier than the HTTP backend but necessary
for single-page applications, dynamically loaded content, and pages that
require client-side JavaScript execution.

Resource lifecycle:
    - On ``__aenter__``: Creates an ``AsyncWebCrawler`` with ``BrowserConfig``
      and calls ``start()`` to launch the Playwright browser process.
    - During scraping: Reuses the SAME browser instance for all pages.
      Each page opens a new tab and closes it after extraction.
    - On ``__aexit__``: Calls ``close()`` to shut down the browser process.
      This is CRITICAL -- failing to close leaks browser processes.

Key crawl4ai classes used:
    - ``AsyncWebCrawler`` (from ``crawl4ai``)
    - ``BrowserConfig`` (from ``crawl4ai``)
    - ``CrawlerRunConfig``, ``CacheMode`` (from ``crawl4ai``)
    - ``LLMExtractionStrategy``, ``LLMConfig`` (from ``crawl4ai``)
    - ``BFSDeepCrawlStrategy``, ``DomainFilter``, ``FilterChain`` (from ``crawl4ai``)

All extraction logic (schema generation, instruction building, result parsing,
merge, completion detection) is in ``scraper.extraction`` -- never duplicated here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, List, Optional

from scraper.config import KeyRotator
from scraper.exceptions import ExtractionError, FetchError
from scraper.extraction import (
    create_extraction_strategy,
    create_run_config,
    get_domain,
    has_list_fields,
    is_complete,
    merge_results,
    normalize_url,
    parse_extraction_result,
    prioritize_urls,
    update_crawl_state,
)
from scraper.models import CrawlState, FieldDefinition, ScraperConfig, ScrapeResult

__all__ = ["BrowserBackend"]

logger = logging.getLogger(__name__)


def _is_key_error(error_msg: str) -> bool:
    """Check if an error message indicates an API key / rate-limit problem.

    Args:
        error_msg: The error message string (lowercased before checking).

    Returns:
        True if the error appears to be a key-related issue (429, 401, auth).
    """
    lower = error_msg.lower()
    return any(
        indicator in lower
        for indicator in ("429", "401", "authentication", "rate limit", "rate_limit", "quota", "unauthorized")
    )


def _make_error_result(url: str, error: str) -> ScrapeResult:
    """Create a ScrapeResult indicating failure.

    Args:
        url: The URL that failed.
        error: Human-readable error description.

    Returns:
        A ScrapeResult with success=False and the error message.
    """
    return ScrapeResult(
        url=url,
        success=False,
        data=None,
        error=error,
    )


def _make_success_result(url: str, data: dict[str, Any]) -> ScrapeResult:
    """Create a ScrapeResult indicating success.

    Args:
        url: The URL that was scraped.
        data: Extracted field data as a dict.

    Returns:
        A ScrapeResult with success=True and the extracted data.
    """
    return ScrapeResult(
        url=url,
        success=True,
        data=data,
        error=None,
    )


class BrowserBackend:
    """Browser-based backend using Playwright for JS-rendered pages.

    Launches a Chromium browser via crawl4ai's Playwright integration.
    The browser instance is reused across all scraping calls within
    the same context manager session.

    Usage::

        async with BrowserBackend(config) as backend:
            result = await backend.scrape_one(url, fields)

    Attributes:
        config: The ScraperConfig controlling behavior.
        _crawler: The crawl4ai AsyncWebCrawler instance (created on enter).
        _key_rotator: Manages API key rotation on LLM failures.
    """

    def __init__(self, config: ScraperConfig) -> None:
        """Initialize the browser backend.

        Args:
            config: ScraperConfig with backend, model, API keys, and other settings.
        """
        self.config = config
        self._crawler: Any = None  # AsyncWebCrawler, set in __aenter__
        self._key_rotator: Optional[KeyRotator] = None

    async def __aenter__(self) -> BrowserBackend:
        """Start the browser backend.

        Creates an ``AsyncWebCrawler`` with a ``BrowserConfig`` and starts
        the Playwright browser process. The browser stays alive until
        ``__aexit__`` is called.

        Returns:
            self

        Raises:
            ConfigError: If no API keys are available.
        """
        from crawl4ai import AsyncWebCrawler, BrowserConfig

        browser_config = BrowserConfig(
            headless=self.config.headless,
            verbose=self.config.verbose,
            enable_stealth=self.config.stealth,
            text_mode=True,  # disables images for faster loads
        )
        self._crawler = AsyncWebCrawler(config=browser_config)
        await self._crawler.start()

        # KeyRotator raises ConfigError if no keys are available
        self._key_rotator = KeyRotator(self.config.api_keys)

        logger.info("BrowserBackend started (browser launched, %d API key(s))", self._key_rotator.total_keys)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shut down the browser backend.

        Closes the Playwright browser process. This MUST be called to
        prevent leaked browser processes. The async context manager
        guarantees this even on errors.
        """
        if self._crawler:
            try:
                await self._crawler.close()
            except Exception:
                logger.warning("Error closing browser", exc_info=True)
            finally:
                self._crawler = None
        logger.info("BrowserBackend shut down")

    # ------------------------------------------------------------------
    # Single-URL mode
    # ------------------------------------------------------------------

    async def scrape_one(
        self,
        url: str,
        fields: List[FieldDefinition],
    ) -> ScrapeResult:
        """Scrape a single URL using the browser and extract fields.

        Identical contract to HTTPBackend.scrape_one. The browser opens
        a new tab for this URL, renders JavaScript, extracts content,
        then closes the tab.

        Args:
            url: The URL to scrape.
            fields: List of FieldDefinition objects describing what to extract.

        Returns:
            A ScrapeResult with extracted data or error information.
        """
        api_key = self._key_rotator.current_key
        return await self._scrape_one_with_key(url, fields, api_key)

    async def _scrape_one_with_key(
        self,
        url: str,
        fields: List[FieldDefinition],
        api_key: str,
        *,
        is_retry: bool = False,
    ) -> ScrapeResult:
        """Internal helper: scrape a URL with a specific API key.

        On key-related errors and ``is_retry=False``, rotates to next key
        and retries once.

        Args:
            url: URL to scrape.
            fields: Field definitions for extraction.
            api_key: The API key to use for this attempt.
            is_retry: If True, do not retry on key errors.

        Returns:
            ScrapeResult with data or error.
        """
        try:
            extraction_strategy = create_extraction_strategy(fields, self.config, api_key)
            run_config = create_run_config(
                extraction_strategy,
                page_timeout=self.config.timeout,
            )

            result = await self._crawler.arun(url=url, config=run_config)

            if not result.success:
                error_msg = result.error_message or "Unknown fetch error"
                logger.warning("Fetch failed for %s: %s", url, error_msg)

                # Check if the error is key-related and we can retry
                if not is_retry and _is_key_error(error_msg):
                    return await self._retry_with_rotated_key(url, fields, error_msg)

                return _make_error_result(url, error_msg)

            # Parse the extracted content (JSON string -> dict)
            if not result.extracted_content:
                logger.warning("No extracted content for %s", url)
                return _make_error_result(url, "No extracted content returned")

            data = parse_extraction_result(result.extracted_content, fields, url)
            self._key_rotator.reset()
            return _make_success_result(url, data)

        except ExtractionError as exc:
            error_msg = str(exc)
            logger.warning("Extraction error for %s: %s", url, error_msg)

            if not is_retry and _is_key_error(error_msg):
                return await self._retry_with_rotated_key(url, fields, error_msg)

            return _make_error_result(url, error_msg)

        except Exception as exc:
            error_msg = str(exc)
            logger.error("Unexpected error scraping %s: %s", url, error_msg, exc_info=True)

            if not is_retry and _is_key_error(error_msg):
                return await self._retry_with_rotated_key(url, fields, error_msg)

            return _make_error_result(url, error_msg)

    async def _retry_with_rotated_key(
        self,
        url: str,
        fields: List[FieldDefinition],
        error_msg: str,
    ) -> ScrapeResult:
        """Rotate API key and retry scraping once.

        Args:
            url: URL to retry.
            fields: Field definitions.
            error_msg: The error reason from the first attempt.

        Returns:
            ScrapeResult from the retry attempt, or error if rotation fails.
        """
        try:
            new_key = self._key_rotator.rotate(error_msg)
            logger.info("Retrying %s with rotated API key", url)
            return await self._scrape_one_with_key(url, fields, new_key, is_retry=True)
        except Exception as rotate_exc:
            logger.error("Key rotation failed: %s", rotate_exc)
            return _make_error_result(url, f"All API keys exhausted: {error_msg}")

    # ------------------------------------------------------------------
    # Multi-URL streaming mode
    # ------------------------------------------------------------------

    async def scrape_many(
        self,
        urls: List[str],
        fields: List[FieldDefinition],
    ) -> AsyncGenerator[ScrapeResult, None]:
        """Scrape multiple URLs with the browser, yielding results as they complete.

        Same contract as HTTPBackend.scrape_many. Uses ``arun_many()``
        with ``stream=True``.

        Browser concurrency is controlled by ``config.max_concurrent``
        via ``semaphore_count`` in the run config.

        Args:
            urls: List of URL strings to scrape.
            fields: List of FieldDefinition objects describing what to extract.

        Yields:
            ScrapeResult for each URL as extraction completes.
        """
        if not urls:
            return

        # Deduplicate URLs while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for u in urls:
            norm = normalize_url(u)
            if norm not in seen:
                seen.add(norm)
                deduped.append(u)

        if not deduped:
            return

        logger.info("scrape_many: %d URLs (%d unique)", len(urls), len(deduped))

        api_key = self._key_rotator.current_key
        extraction_strategy = create_extraction_strategy(fields, self.config, api_key)

        # Create run config directly with semaphore_count for concurrency control
        from crawl4ai import CacheMode, CrawlerRunConfig

        run_config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS,
            stream=True,
            semaphore_count=self.config.max_concurrent,
            page_timeout=self.config.timeout,
        )

        try:
            async_gen = await self._crawler.arun_many(urls=deduped, config=run_config)

            async for result in async_gen:
                yield self._process_crawl_result(result, fields)

        except Exception as exc:
            logger.error("Error in scrape_many stream: %s", exc, exc_info=True)
            # Yield an error result for any remaining unprocessed state
            yield _make_error_result(
                deduped[0] if deduped else "unknown",
                f"Stream error: {exc}",
            )

    # ------------------------------------------------------------------
    # Crawl mode (BFS deep crawl with completion detection)
    # ------------------------------------------------------------------

    async def crawl(
        self,
        start_url: str,
        fields: List[FieldDefinition],
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> AsyncGenerator[ScrapeResult, None]:
        """Crawl with browser rendering, following links and extracting from each page.

        Same contract as HTTPBackend.crawl. Uses ``BFSDeepCrawlStrategy``
        with streaming and completion detection.

        The browser backend is useful for crawling JavaScript-heavy sites
        where links are dynamically generated and content is client-rendered.

        Args:
            start_url: The starting URL for the crawl.
            fields: List of FieldDefinition objects describing what to extract.
            max_pages: Override for config.max_pages (optional).
            max_depth: Override for config.max_depth (optional).

        Yields:
            ScrapeResult for each crawled page as extraction completes.
        """
        from crawl4ai import (
            BFSDeepCrawlStrategy,
            CacheMode,
            CrawlerRunConfig,
            DomainFilter,
            FilterChain,
        )

        domain = get_domain(start_url)
        effective_max_pages = max_pages or self.config.max_pages
        effective_max_depth = max_depth or self.config.max_depth

        logger.info(
            "Starting crawl: url=%s, domain=%s, max_pages=%d, max_depth=%d",
            start_url,
            domain,
            effective_max_pages,
            effective_max_depth,
        )

        bfs_strategy = BFSDeepCrawlStrategy(
            max_depth=effective_max_depth,
            max_pages=effective_max_pages,
            filter_chain=FilterChain([DomainFilter(allowed_domains=[domain])]),
        )

        api_key = self._key_rotator.current_key
        extraction_strategy = create_extraction_strategy(fields, self.config, api_key)

        run_config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            deep_crawl_strategy=bfs_strategy,
            cache_mode=CacheMode.BYPASS,
            stream=True,
            page_timeout=self.config.timeout,
        )

        state = CrawlState()
        _has_list = has_list_fields(fields)

        try:
            crawl_output = await self._crawler.arun(url=start_url, config=run_config)

            # crawl4ai ContextVar bug: after bfs_strategy.shutdown() corrupts
            # deep_crawl_active, subsequent calls may return a list/container
            # instead of an async generator. Handle both cases.
            if hasattr(crawl_output, "__aiter__"):
                result_iter = crawl_output.__aiter__()
            else:
                # Fallback: wrap synchronous iterable as async
                async def _wrap_sync(items):
                    for item in items:
                        yield item
                result_iter = _wrap_sync(crawl_output).__aiter__()

            async for result in result_iter:
                # Process each crawl result into a ScrapeResult
                if not result.success:
                    error_msg = result.error_message or "Crawl fetch failed"
                    logger.warning("Crawl page failed (%s): %s", result.url, error_msg)
                    yield _make_error_result(result.url, error_msg)
                    continue

                # Parse extracted content
                page_data = self._parse_result_safe(result, fields)
                if page_data is not None:
                    # Update crawl state with merged data
                    update_crawl_state(state, page_data, fields, result.url)
                    yield _make_success_result(result.url, page_data)
                else:
                    # Extraction failed or no content; still count the page visit
                    state.urls_visited.add(normalize_url(result.url))
                    state.pages_scraped += 1
                    yield _make_error_result(result.url, "No extractable content")

                # Completion detection: stop early if all scalar fields are filled
                # and there are no list fields
                if state.is_complete and not _has_list:
                    state.completion_reason = "all_fields_found"
                    logger.info(
                        "Crawl complete: all scalar fields found after %d pages",
                        state.pages_scraped,
                    )
                    try:
                        await bfs_strategy.shutdown()
                    except Exception:
                        logger.debug("BFS shutdown raised (may already be done)", exc_info=True)
                    break

        except Exception as exc:
            logger.error("Error in crawl stream: %s", exc, exc_info=True)
            state.completion_reason = "error"
            yield _make_error_result(start_url, f"Crawl error: {exc}")

        # Log final crawl summary
        if state.completion_reason is None:
            if state.pages_scraped >= effective_max_pages:
                state.completion_reason = "max_pages_reached"
            else:
                state.completion_reason = "no_more_links"

        logger.info(
            "Crawl finished: %d pages scraped, reason=%s, complete=%s",
            state.pages_scraped,
            state.completion_reason,
            state.is_complete,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_crawl_result(
        self,
        result: Any,
        fields: List[FieldDefinition],
    ) -> ScrapeResult:
        """Convert a single crawl4ai CrawlResult into a ScrapeResult.

        Handles both success and failure cases without raising exceptions.

        Args:
            result: A crawl4ai CrawlResult object.
            fields: Field definitions for parsing.

        Returns:
            A ScrapeResult with either extracted data or error info.
        """
        url = getattr(result, "url", "unknown")

        if not result.success:
            error_msg = result.error_message or "Fetch failed"
            logger.warning("Fetch failed for %s: %s", url, error_msg)
            return _make_error_result(url, error_msg)

        page_data = self._parse_result_safe(result, fields)
        if page_data is not None:
            self._key_rotator.reset()
            return _make_success_result(url, page_data)
        else:
            return _make_error_result(url, "No extractable content")

    def _parse_result_safe(
        self,
        result: Any,
        fields: List[FieldDefinition],
    ) -> Optional[dict[str, Any]]:
        """Safely parse extracted_content from a crawl4ai result.

        Returns None if there is no content or parsing fails, instead of
        raising an exception.

        Args:
            result: A crawl4ai CrawlResult with extracted_content.
            fields: Field definitions for the parse_extraction_result call.

        Returns:
            A parsed data dict, or None if parsing fails.
        """
        url = getattr(result, "url", "unknown")

        if not result.extracted_content:
            logger.debug("No extracted_content for %s", url)
            return None

        try:
            return parse_extraction_result(result.extracted_content, fields, url)
        except ExtractionError as exc:
            logger.warning("Failed to parse extraction for %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected parse error for %s: %s", url, exc)
            return None
