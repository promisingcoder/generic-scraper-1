"""Shared extraction module -- the heart of the scraper library.

This module contains ALL extraction-related logic used by both backends.
Nothing here is duplicated. Both ``http_backend`` and ``browser_backend``
import from this module for:

- **Schema generation**: Building JSON schemas from FieldDefinition lists
- **Instruction building**: Constructing LLM prompts from field definitions
- **Strategy creation**: Creating crawl4ai LLMExtractionStrategy and CrawlerRunConfig
- **Result parsing**: Parsing and validating LLM extraction output
- **Completion detection**: Determining if all scalar fields are filled
- **Result merging**: Combining extraction results across multiple pages
- **URL normalization**: Deduplicating URLs during crawl
- **API key management**: Getting the current API key with rotation support

Design principles:
- Pure functions where possible (no hidden state)
- The only external dependency is crawl4ai (for strategy/config creation)
- All other functions are pure Python/Pydantic logic
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from scraper.config import KeyRotator
from scraper.exceptions import ConfigError, ExtractionError
from scraper.models import (
    CrawlState,
    FieldDefinition,
    ScraperConfig,
    ScrapeResult,
)

__all__ = [
    "build_json_schema",
    "build_instruction",
    "create_extraction_strategy",
    "create_run_config",
    "parse_extraction_result",
    "has_list_fields",
    "is_complete",
    "merge_results",
    "normalize_url",
    "get_domain",
    "prioritize_urls",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field type to JSON Schema type mapping
# ---------------------------------------------------------------------------

_FIELD_TYPE_TO_JSON_SCHEMA: Dict[str, Dict[str, Any]] = {
    "string": {"type": "string"},
    "number": {"type": "number"},
    "integer": {"type": "integer"},
    "boolean": {"type": "boolean"},
    "list[string]": {"type": "array", "items": {"type": "string"}},
    "list[number]": {"type": "array", "items": {"type": "number"}},
    "list[boolean]": {"type": "array", "items": {"type": "boolean"}},
    "list[object]": {"type": "array", "items": {"type": "object"}},
}


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


def build_json_schema(fields: List[FieldDefinition]) -> Dict[str, Any]:
    """Build a JSON Schema dict from a list of FieldDefinitions.

    The generated schema is suitable for passing to crawl4ai's
    ``LLMExtractionStrategy(schema=...)`` parameter. It describes an object
    whose properties correspond to the requested fields.

    Args:
        fields: List of FieldDefinition objects describing what to extract.

    Returns:
        A JSON Schema dict with ``type: "object"`` and properties for each
        field. No ``required`` constraint is set -- all fields are optional
        so the LLM can return null for fields it cannot find.

    Raises:
        ConfigError: If the fields list is empty.

    Examples:
        >>> from scraper.models import FieldDefinition
        >>> fields = [
        ...     FieldDefinition(name="company", description="Company name", type="string"),
        ...     FieldDefinition(name="emails", description="Contact emails", type="list[string]"),
        ... ]
        >>> schema = build_json_schema(fields)
        >>> schema["properties"]["company"]
        {'type': 'string', 'description': 'Company name'}
    """
    if not fields:
        raise ConfigError("At least one field definition is required")

    properties: Dict[str, Any] = {}
    for field in fields:
        json_type = _FIELD_TYPE_TO_JSON_SCHEMA.get(field.type)
        if json_type is None:
            raise ConfigError(f"Unsupported field type: {field.type!r}")

        prop = dict(json_type)  # shallow copy
        prop["description"] = field.description
        properties[field.name] = prop

    return {
        "type": "object",
        "properties": properties,
        "description": "Extracted data from the web page",
    }


# ---------------------------------------------------------------------------
# Instruction building
# ---------------------------------------------------------------------------


def build_instruction(fields: List[FieldDefinition]) -> str:
    """Build a clear LLM instruction string from field definitions.

    The instruction tells the LLM exactly what fields to extract, their
    expected types, and how to handle missing data. This is passed to
    ``LLMExtractionStrategy(instruction=...)``.

    Args:
        fields: List of FieldDefinition objects.

    Returns:
        A multi-line instruction string for the LLM.

    Raises:
        ConfigError: If the fields list is empty.

    Examples:
        >>> fields = [FieldDefinition(name="name", description="Company name", type="string")]
        >>> print(build_instruction(fields))  # doctest: +SKIP
        Extract the following fields from the page content...
    """
    if not fields:
        raise ConfigError("At least one field definition is required")

    lines = [
        "Extract the following fields from the page content.",
        "Return the result as a single JSON object with exactly these keys.",
        "If a field cannot be found on the page, set its value to null.",
        "Do NOT invent or hallucinate data that is not present on the page.",
        "",
        "Fields to extract:",
    ]

    for field in fields:
        type_hint = f" (type: {field.type})" if field.type != "string" else ""
        lines.append(f'  - "{field.name}": {field.description}{type_hint}')

    lines.extend([
        "",
        "Return ONLY the JSON object, no additional text or explanation.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# crawl4ai strategy and config creation
# ---------------------------------------------------------------------------


def create_extraction_strategy(
    fields: List[FieldDefinition],
    config: ScraperConfig,
    api_key: str,
) -> Any:
    """Create a crawl4ai LLMExtractionStrategy for the given fields and config.

    This function imports crawl4ai classes and creates a properly configured
    extraction strategy. It is the ONLY place in the codebase where
    LLMExtractionStrategy is instantiated.

    Args:
        fields: List of FieldDefinition objects to extract.
        config: ScraperConfig with LLM model and parameters.
        api_key: The API key to use for LLM calls.

    Returns:
        An ``LLMExtractionStrategy`` instance ready for use in a CrawlerRunConfig.

    Raises:
        ConfigError: If required configuration is missing.
    """
    from crawl4ai import LLMConfig, LLMExtractionStrategy

    schema = build_json_schema(fields)
    instruction = build_instruction(fields)

    llm_config = LLMConfig(
        provider=config.model,
        api_token=api_key,
    )

    strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        instruction=instruction,
        schema=schema,
        # extraction_type is auto-set to "schema" because schema is provided
        input_format=config.input_format,
        chunk_token_threshold=config.chunk_token_threshold,
        verbose=config.verbose,
    )

    return strategy


def create_run_config(
    extraction_strategy: Any,
    *,
    stream: bool = False,
    page_timeout: int = 30000,
) -> Any:
    """Create a crawl4ai CrawlerRunConfig with the given extraction strategy.

    Args:
        extraction_strategy: An LLMExtractionStrategy instance.
        stream: Whether to enable streaming mode (for arun_many / deep crawl).
        page_timeout: Page load timeout in milliseconds.

    Returns:
        A ``CrawlerRunConfig`` instance ready for use with ``arun()`` or ``arun_many()``.
    """
    from crawl4ai import CacheMode, CrawlerRunConfig

    return CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.BYPASS,
        stream=stream,
        page_timeout=page_timeout,
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_extraction_result(
    raw: Optional[str],
    fields: List[FieldDefinition],
    url: str = "",
) -> Dict[str, Any]:
    """Parse the JSON string from crawl4ai's extracted_content into a dict.

    crawl4ai returns ``extracted_content`` as a JSON string. This function
    parses it, extracts the relevant fields, and normalizes the result
    to always contain all requested field names (with None for missing ones).

    The LLM may return a JSON array (list of objects) or a single object.
    If an array is returned, the first element is used. If the first element
    itself is a list, we attempt to find the first dict element.

    Args:
        raw: The raw JSON string from ``result.extracted_content``.
        fields: The field definitions that were requested.
        url: The source URL (for error context).

    Returns:
        A dict with keys matching field names. Missing fields have None values.

    Raises:
        ExtractionError: If the JSON cannot be parsed or contains no usable data.

    Examples:
        >>> fields = [FieldDefinition(name="title", description="Page title", type="string")]
        >>> parse_extraction_result('{"title": "Hello"}', fields)
        {'title': 'Hello'}
        >>> parse_extraction_result('[{"title": "Hello"}]', fields)
        {'title': 'Hello'}
    """
    if not raw or raw.strip() == "":
        raise ExtractionError(
            "Empty extraction result",
            url=url,
            raw_response=raw,
        )

    # Parse JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ExtractionError(
            f"Failed to parse extraction result as JSON: {e}",
            url=url,
            raw_response=raw,
        ) from e

    # Unwrap: if it's a list, take the first dict element
    data = _unwrap_to_dict(parsed, url=url, raw=raw)

    # Build normalized result with all requested field names
    result: Dict[str, Any] = {}
    for field in fields:
        value = data.get(field.name)
        result[field.name] = _normalize_value(value, field)

    return result


def _unwrap_to_dict(
    parsed: Any,
    url: str = "",
    raw: str = "",
) -> Dict[str, Any]:
    """Unwrap parsed JSON to a single dict.

    Handles these cases:
    - Already a dict: return as-is
    - A list containing dicts: return the first dict
    - A list containing a single list of dicts: unwrap one more level
    - Anything else: raise ExtractionError
    """
    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, list):
        if not parsed:
            raise ExtractionError(
                "Extraction returned an empty list",
                url=url,
                raw_response=raw,
            )

        # Find the first dict in the list
        for item in parsed:
            if isinstance(item, dict):
                return item

        # If first element is itself a list, unwrap one more level
        if isinstance(parsed[0], list):
            for item in parsed[0]:
                if isinstance(item, dict):
                    return item

        raise ExtractionError(
            "Extraction returned a list with no dict elements",
            url=url,
            raw_response=raw,
        )

    raise ExtractionError(
        f"Unexpected extraction result type: {type(parsed).__name__}",
        url=url,
        raw_response=raw,
    )


def _normalize_value(value: Any, field: FieldDefinition) -> Any:
    """Normalize an extracted value to match the expected field type.

    - For scalar fields: returns the value as-is, or None if missing/empty
    - For list fields: ensures the value is a list, wraps single values
    """
    if value is None:
        return [] if field.is_list else None

    if field.is_list:
        if isinstance(value, list):
            return value
        # Single value -- wrap in list
        return [value]

    return value


# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------


def has_list_fields(fields: List[FieldDefinition]) -> bool:
    """Check if any field in the list is a list type.

    This determines whether crawl completion can ever be achieved by
    filling all fields. If any list field exists, crawling must continue
    until ``max_pages`` is reached or no more links remain.

    Args:
        fields: List of FieldDefinition objects.

    Returns:
        True if any field has a type starting with ``"list["``; False otherwise.

    Examples:
        >>> from scraper.models import FieldDefinition
        >>> has_list_fields([FieldDefinition(name="name", description="Name", type="string")])
        False
        >>> has_list_fields([FieldDefinition(name="emails", description="Emails", type="list[string]")])
        True
    """
    return any(field.is_list for field in fields)


def is_complete(merged_data: Dict[str, Any], fields: List[FieldDefinition]) -> bool:
    """Determine if all scalar fields have been filled with non-null values.

    Completion detection rules:
    1. If ANY field is a list type, return False (lists are never "complete"
       because we cannot know the total count).
    2. If ALL fields are scalar and ALL have non-null values, return True.
    3. Otherwise, return False (some scalar fields are still missing).

    This function is called after each page in crawl mode to decide
    whether to stop early.

    Args:
        merged_data: The accumulated extraction data dict.
        fields: The field definitions that were requested.

    Returns:
        True if crawling can stop because all data has been found.

    Examples:
        >>> from scraper.models import FieldDefinition
        >>> fields = [FieldDefinition(name="name", description="Name", type="string")]
        >>> is_complete({"name": "Acme Corp"}, fields)
        True
        >>> is_complete({"name": None}, fields)
        False
        >>> fields_with_list = fields + [FieldDefinition(name="emails", description="Emails", type="list[string]")]
        >>> is_complete({"name": "Acme Corp", "emails": ["a@b.com"]}, fields_with_list)
        False
    """
    if has_list_fields(fields):
        return False

    for field in fields:
        if not field.is_scalar:
            continue
        value = merged_data.get(field.name)
        if value is None:
            return False

    return True


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------


def merge_results(
    existing: Dict[str, Any],
    new: Dict[str, Any],
    fields: List[FieldDefinition],
) -> Dict[str, Any]:
    """Merge two extraction results according to field type rules.

    Merging rules:
    - **Scalar fields** (string, number, boolean, integer): Keep the first
      non-null value. If ``existing`` already has a non-null value, the
      ``new`` value is ignored.
    - **List fields** (list[string], list[number], etc.): Union the two
      lists and deduplicate. For ``list[object]``, deduplication uses
      JSON serialization for comparison.

    This function is used in crawl mode to accumulate results across
    multiple pages.

    Args:
        existing: The current accumulated data dict.
        new: The newly extracted data dict from the latest page.
        fields: The field definitions for type information.

    Returns:
        A new dict containing the merged result.

    Examples:
        >>> from scraper.models import FieldDefinition
        >>> fields = [
        ...     FieldDefinition(name="name", description="Name", type="string"),
        ...     FieldDefinition(name="emails", description="Emails", type="list[string]"),
        ... ]
        >>> merge_results({"name": None, "emails": ["a@b.com"]}, {"name": "Acme", "emails": ["c@d.com"]}, fields)
        {'name': 'Acme', 'emails': ['a@b.com', 'c@d.com']}
        >>> merge_results({"name": "Acme", "emails": []}, {"name": "Other", "emails": ["x@y.com"]}, fields)
        {'name': 'Acme', 'emails': ['x@y.com']}
    """
    merged: Dict[str, Any] = {}

    for field in fields:
        existing_val = existing.get(field.name)
        new_val = new.get(field.name)

        if field.is_list:
            merged[field.name] = _merge_lists(
                existing_val if isinstance(existing_val, list) else [],
                new_val if isinstance(new_val, list) else [],
                field,
            )
        else:
            # Scalar: keep first non-null
            merged[field.name] = existing_val if existing_val is not None else new_val

    return merged


def _merge_lists(
    existing: List[Any],
    new: List[Any],
    field: FieldDefinition,
) -> List[Any]:
    """Merge two lists with deduplication.

    For list[object], deduplication uses JSON serialization of each item.
    For primitive list types, uses direct equality.
    """
    if not new:
        return list(existing)
    if not existing:
        return list(new)

    if field.type == "list[object]":
        # Use JSON serialization for dedup of complex objects
        seen: set[str] = set()
        result: List[Any] = []
        for item in existing + new:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
    else:
        # Primitive types -- direct dedup preserving order
        seen_values: set[Any] = set()
        result = []
        for item in existing + new:
            # Convert unhashable types to their string repr for dedup
            try:
                hashable_key = item
                if hashable_key not in seen_values:
                    seen_values.add(hashable_key)
                    result.append(item)
            except TypeError:
                # Unhashable item, fall back to JSON key
                str_key = json.dumps(item, sort_keys=True, default=str)
                if str_key not in seen_values:
                    seen_values.add(str_key)
                    result.append(item)
        return result


# ---------------------------------------------------------------------------
# URL normalization and domain utilities
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication purposes.

    Normalization steps:
    1. Parse the URL into components
    2. Convert scheme and host to lowercase
    3. Remove default ports (80 for http, 443 for https)
    4. Remove fragment (``#section``)
    5. Remove trailing slash from path (unless path is just ``/``)
    6. Sort query parameters alphabetically
    7. Reassemble the URL

    Args:
        url: The URL string to normalize.

    Returns:
        The normalized URL string.

    Examples:
        >>> normalize_url("https://Example.COM/about/#section")
        'https://example.com/about'
        >>> normalize_url("https://example.com/page?b=2&a=1")
        'https://example.com/page?a=1&b=2'
        >>> normalize_url("https://example.com/")
        'https://example.com/'
    """
    parsed = urlparse(url)

    # Lowercase scheme and host
    scheme = parsed.scheme.lower()
    netloc = parsed.hostname or ""
    netloc = netloc.lower()

    # Handle port -- remove default ports
    port = parsed.port
    if port:
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
    if port:
        netloc = f"{netloc}:{port}"

    # Remove fragment
    fragment = ""

    # Normalize path -- remove trailing slash (keep root "/")
    path = parsed.path
    if path and path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Sort query parameters
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    sorted_query = urlencode(
        sorted(
            [(k, v[0] if len(v) == 1 else v) for k, v in query_params.items()]
        ),
        doseq=True,
    )

    return urlunparse((scheme, netloc, path, parsed.params, sorted_query, fragment))


def get_domain(url: str) -> str:
    """Extract the domain (netloc) from a URL.

    Args:
        url: A URL string.

    Returns:
        The lowercase domain, e.g., ``"www.example.com"``.

    Examples:
        >>> get_domain("https://www.Example.COM/page")
        'www.example.com'
    """
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


# ---------------------------------------------------------------------------
# URL prioritization for crawl mode
# ---------------------------------------------------------------------------

# High-value path segments that often contain useful information
_HIGH_VALUE_PATTERNS: List[str] = [
    "/about",
    "/contact",
    "/team",
    "/staff",
    "/people",
    "/services",
    "/leadership",
    "/attorneys",
    "/lawyers",
    "/professionals",
    "/our-team",
    "/our-people",
    "/who-we-are",
    "/company",
    "/partners",
    "/directors",
    "/management",
    "/board",
    "/overview",
    "/practice",
    "/areas",
]


def prioritize_urls(urls: List[str]) -> List[str]:
    """Sort URLs so that high-value paths come first.

    High-value paths are those likely to contain important information
    such as ``/about``, ``/contact``, ``/team``, ``/staff``, etc.
    These are sorted to the front of the list.

    Args:
        urls: List of URL strings to prioritize.

    Returns:
        A new list with high-value URLs first, then remaining URLs
        in their original order.

    Examples:
        >>> prioritize_urls(["https://x.com/blog", "https://x.com/about", "https://x.com/team"])
        ['https://x.com/about', 'https://x.com/team', 'https://x.com/blog']
    """

    def score(url: str) -> int:
        path = urlparse(url).path.lower()
        for i, pattern in enumerate(_HIGH_VALUE_PATTERNS):
            if pattern in path:
                return i  # Lower index = higher priority
        return len(_HIGH_VALUE_PATTERNS) + 1  # Non-matching URLs go last

    return sorted(urls, key=score)


# ---------------------------------------------------------------------------
# CrawlState helpers
# ---------------------------------------------------------------------------


def update_crawl_state(
    state: CrawlState,
    new_data: Dict[str, Any],
    fields: List[FieldDefinition],
    url: str,
) -> CrawlState:
    """Update a CrawlState with new extraction results from a page.

    This mutates the state in-place for efficiency (CrawlState is not frozen).

    Args:
        state: The current crawl state to update.
        new_data: Extracted data dict from the latest page.
        fields: The field definitions for merge logic.
        url: The URL that was just crawled (will be normalized and added to visited set).

    Returns:
        The same state object, mutated with new data.
    """
    normalized = normalize_url(url)
    state.urls_visited.add(normalized)
    state.pages_scraped += 1

    if new_data:
        state.merged_data = merge_results(state.merged_data, new_data, fields)
        state.is_complete = is_complete(state.merged_data, fields)

    return state
