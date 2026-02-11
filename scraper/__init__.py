"""scraper -- A web scraper library with LLM-based extraction.

Quick start::

    from scraper import Scraper, define_fields

    fields = define_fields(
        ("company_name", "The name of the company"),
        ("emails", "All contact email addresses", "list[string]"),
    )

    # Async usage (recommended)
    async with Scraper() as s:
        result = await s.scrape("https://example.com", fields)
        print(result.data)

    # Sync usage (simple scripts)
    from scraper import scrape_sync
    result = scrape_sync("https://example.com", fields)

Three modes of operation:

1. **Single-URL** (``scraper.scrape()``): Scrape one page, extract, return.
2. **Multi-URL** (``scraper.scrape_many()``): Scrape multiple URLs concurrently,
   yield results as each completes.
3. **Crawl** (``scraper.crawl()``): Follow links within a site, extract from
   each page, yield results incrementally with completion detection.

Two fetching backends:

- **HTTP** (default): Fast, lightweight, for static pages.
- **Browser**: Playwright-based, for JavaScript-rendered pages.
"""

from scraper.api import Scraper, define_fields, scrape_sync
from scraper.config import KeyRotator, load_config
from scraper.exceptions import (
    AllKeysExhaustedError,
    ConfigError,
    ExtractionError,
    FetchError,
    ScraperError,
)
from scraper.models import (
    CrawlState,
    FieldDefinition,
    ScraperConfig,
    ScrapeResult,
)

__all__ = [
    # Primary API
    "Scraper",
    "define_fields",
    "scrape_sync",
    # Models
    "FieldDefinition",
    "ScrapeResult",
    "CrawlState",
    "ScraperConfig",
    # Config
    "load_config",
    "KeyRotator",
    # Exceptions
    "ScraperError",
    "FetchError",
    "ExtractionError",
    "ConfigError",
    "AllKeysExhaustedError",
]

__version__ = "0.1.0"
