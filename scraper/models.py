"""Pydantic data models for the scraper library.

This module defines the core data structures used throughout the library:

- **FieldDefinition**: Describes a single field the user wants to extract.
- **ScrapeResult**: The result of extracting data from a single URL.
- **CrawlState**: Internal mutable state tracking during a crawl session.
- **ScraperConfig**: Full configuration for the scraper (backend, LLM, concurrency, etc.).

All models use Pydantic v2 with strict validation where appropriate.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "FieldDefinition",
    "ScrapeResult",
    "CrawlState",
    "ScraperConfig",
    "SCALAR_TYPES",
    "LIST_TYPE_PREFIX",
]

# ---------------------------------------------------------------------------
# Constants for field type classification
# ---------------------------------------------------------------------------

SCALAR_TYPES: frozenset[str] = frozenset({"string", "number", "boolean", "integer"})
"""Field types that represent single scalar values."""

LIST_TYPE_PREFIX: str = "list["
"""Prefix used to identify list-type fields (e.g., 'list[string]', 'list[object]')."""


# ---------------------------------------------------------------------------
# FieldDefinition
# ---------------------------------------------------------------------------


class FieldDefinition(BaseModel):
    """Describes a single field to extract from a web page.

    Attributes:
        name: The field name used as a key in the extracted data dict.
              Must be a valid Python identifier (e.g., ``"company_name"``).
        description: A human-readable description telling the LLM what to
                     extract (e.g., ``"The official name of the company"``).
        type: The expected data type. Scalar types (``"string"``, ``"number"``,
              ``"boolean"``, ``"integer"``) are used for single values. List
              types (``"list[string]"``, ``"list[number]"``, ``"list[object]"``,
              ``"list[boolean]"``) are used for collections. The type affects
              completion detection in crawl mode: scalar fields can be
              "completed" when a non-null value is found, but list fields
              are never considered complete since the total count is unknown.

    Examples:
        >>> FieldDefinition(name="company_name", description="Name of the company", type="string")
        >>> FieldDefinition(name="emails", description="All contact emails", type="list[string]")
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Field name used as key in extracted data (e.g., 'company_name')",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of what to extract",
    )
    type: str = Field(
        default="string",
        description=(
            "Expected data type: 'string', 'number', 'boolean', 'integer', "
            "or list types like 'list[string]', 'list[number]', 'list[object]', 'list[boolean]'"
        ),
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the type string is one of the supported types."""
        allowed_scalars = {"string", "number", "boolean", "integer"}
        allowed_lists = {"list[string]", "list[number]", "list[object]", "list[boolean]"}
        allowed = allowed_scalars | allowed_lists
        if v not in allowed:
            raise ValueError(
                f"Invalid field type '{v}'. Must be one of: {sorted(allowed)}"
            )
        return v

    @property
    def is_list(self) -> bool:
        """Return True if this field is a list type."""
        return self.type.startswith(LIST_TYPE_PREFIX)

    @property
    def is_scalar(self) -> bool:
        """Return True if this field is a scalar type."""
        return self.type in SCALAR_TYPES


# ---------------------------------------------------------------------------
# ScrapeResult
# ---------------------------------------------------------------------------


class ScrapeResult(BaseModel):
    """Result of extracting data from a single URL.

    Returned by ``scrape()``, yielded by ``scrape_many()`` and ``crawl()``.

    Attributes:
        url: The URL that was scraped.
        success: Whether extraction succeeded without errors.
        data: Extracted fields as a key-value dict. Keys match the field
              names from the FieldDefinition list. Values are the extracted
              data, or None for fields that could not be found.
        error: Error message if extraction failed, None otherwise.
        timestamp: ISO 8601 timestamp of when the result was produced.
    """

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., description="The URL that was scraped")
    success: bool = Field(..., description="Whether extraction succeeded")
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted fields as key-value pairs, or None on failure",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if extraction failed",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of result creation",
    )


# ---------------------------------------------------------------------------
# CrawlState
# ---------------------------------------------------------------------------


class CrawlState(BaseModel):
    """Mutable internal state for tracking progress during a crawl session.

    This model is NOT part of the public API. It is used internally by the
    crawl engine to accumulate results across multiple pages and determine
    when crawling should stop.

    Attributes:
        merged_data: Accumulated extraction results merged across all crawled
                     pages. Scalar fields keep the first non-null value found;
                     list fields are unioned and deduplicated.
        pages_scraped: Total number of pages successfully scraped so far.
        urls_visited: Set of normalized URLs that have been visited (for dedup).
        is_complete: Whether all scalar fields have non-null values. When True
                     and there are no list fields, crawling can stop early.
        completion_reason: The reason crawling stopped, set when crawling ends.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    merged_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated merged extraction results",
    )
    pages_scraped: int = Field(
        default=0,
        ge=0,
        description="Number of pages successfully scraped",
    )
    urls_visited: Set[str] = Field(
        default_factory=set,
        description="Set of normalized URLs already visited",
    )
    is_complete: bool = Field(
        default=False,
        description="Whether all scalar fields have been found",
    )
    completion_reason: Optional[
        Literal["all_fields_found", "max_pages_reached", "no_more_links", "error"]
    ] = Field(
        default=None,
        description="Reason crawling stopped",
    )


# ---------------------------------------------------------------------------
# ScraperConfig
# ---------------------------------------------------------------------------


class ScraperConfig(BaseModel):
    """Configuration for the scraper.

    Controls the backend selection, LLM provider, concurrency limits,
    timeouts, and crawl-mode behavior. Sensible defaults are provided
    for all fields.

    API keys are loaded from environment variables if not provided
    explicitly. The primary key comes from ``OPENAI_API_KEY``, with
    optional backups from ``OPENAI_API_KEY_BACKUP_1``,
    ``OPENAI_API_KEY_BACKUP_2``, etc.

    Attributes:
        backend: Which fetching backend to use. ``"http"`` for fast
                 static page fetching, ``"browser"`` for JS-rendered pages.
        model: LLM model identifier in litellm format
               (e.g., ``"openai/gpt-4o-mini"``).
        api_keys: List of API keys. The first is primary; others are
                  backups for rotation on 429/auth errors.
        max_concurrent: Maximum concurrent requests for multi-URL mode.
                        Default 5 for HTTP, recommended 3 for browser.
        timeout: Page load timeout in milliseconds.
        headless: Whether to run browser in headless mode (browser backend only).
        stealth: Whether to enable anti-bot stealth mode (browser backend only).
        max_pages: Maximum pages to crawl in crawl mode before stopping.
        max_depth: Maximum link-following depth in crawl mode.
        input_format: Content format sent to the LLM for extraction.
                      One of ``"markdown"``, ``"html"``, ``"fit_markdown"``,
                      ``"cleaned_html"``, ``"fit_html"``.
        chunk_token_threshold: Maximum tokens per chunk sent to the LLM.
        verbose: Whether to enable verbose logging from crawl4ai.
    """

    model_config = ConfigDict(validate_default=True)

    backend: Literal["http", "browser"] = Field(
        default="http",
        description="Fetching backend: 'http' for static pages, 'browser' for JS-rendered",
    )
    model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model in litellm format (e.g., 'openai/gpt-4o-mini')",
    )
    api_keys: List[str] = Field(
        default_factory=list,
        description="API keys for the LLM provider; first is primary, rest are backups",
    )
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent requests (5 for HTTP, 3 recommended for browser)",
    )
    timeout: int = Field(
        default=30000,
        ge=1000,
        description="Page load timeout in milliseconds",
    )
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode (browser backend only)",
    )
    stealth: bool = Field(
        default=True,
        description="Enable anti-bot stealth mode (browser backend only)",
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        description="Maximum pages to crawl in crawl mode",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        description="Maximum link-following depth in crawl mode",
    )
    input_format: Literal[
        "markdown", "html", "fit_markdown", "cleaned_html", "fit_html"
    ] = Field(
        default="markdown",
        description="Content format sent to the LLM for extraction",
    )
    chunk_token_threshold: int = Field(
        default=2048,
        ge=256,
        description="Maximum tokens per chunk for LLM extraction",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging from crawl4ai internals",
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def load_api_keys_from_env(cls, v: List[str]) -> List[str]:
        """If no API keys provided, load them from environment variables.

        Checks OPENAI_API_KEY, then OPENAI_API_KEY_BACKUP_1,
        OPENAI_API_KEY_BACKUP_2, etc.
        """
        if v:
            return v

        keys: List[str] = []
        primary = os.environ.get("OPENAI_API_KEY")
        if primary:
            keys.append(primary)

        # Check for numbered backups
        for i in range(1, 10):
            backup = os.environ.get(f"OPENAI_API_KEY_BACKUP_{i}")
            if backup:
                keys.append(backup)
            else:
                break  # Stop at the first missing backup

        return keys
