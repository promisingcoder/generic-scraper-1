"""Exception hierarchy for the scraper library.

All exceptions inherit from ScraperError, which inherits from Exception.
This allows callers to catch all scraper-related errors with a single
``except ScraperError`` clause, or to catch specific categories.

Hierarchy::

    ScraperError
    +-- FetchError          -- page fetch failures (HTTP errors, timeouts, connection)
    +-- ExtractionError     -- LLM extraction failures (bad JSON, API errors)
    +-- ConfigError         -- invalid configuration (bad backend name, missing fields)
    +-- AllKeysExhaustedError -- all API keys have been tried and failed
"""

from __future__ import annotations

__all__ = [
    "ScraperError",
    "FetchError",
    "ExtractionError",
    "ConfigError",
    "AllKeysExhaustedError",
]


class ScraperError(Exception):
    """Base exception for all scraper library errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context (URL, status code, etc.).
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class FetchError(ScraperError):
    """Raised when a page cannot be fetched.

    Covers HTTP errors, connection timeouts, DNS failures, and any other
    network-level issue that prevents the page content from being retrieved.

    Examples:
        - HTTP 404 or 500 responses
        - Connection timeout after configured timeout period
        - DNS resolution failure
        - SSL certificate errors
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
        details: dict | None = None,
    ) -> None:
        combined = dict(details or {})
        if url is not None:
            combined["url"] = url
        if status_code is not None:
            combined["status_code"] = status_code
        super().__init__(message, combined)
        self.url = url
        self.status_code = status_code


class ExtractionError(ScraperError):
    """Raised when LLM-based extraction fails.

    Covers cases where the LLM returns invalid JSON, the API call fails,
    the response does not match the expected schema, or the extraction
    produces no usable data.

    Examples:
        - LLM returns malformed JSON
        - OpenAI API returns a 500 error
        - Extracted content does not match the requested schema
        - Empty extraction result when data was expected
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        raw_response: str | None = None,
        details: dict | None = None,
    ) -> None:
        combined = dict(details or {})
        if url is not None:
            combined["url"] = url
        if raw_response is not None:
            combined["raw_response"] = raw_response[:500]  # Truncate for readability
        super().__init__(message, combined)
        self.url = url
        self.raw_response = raw_response


class ConfigError(ScraperError):
    """Raised when the scraper configuration is invalid.

    Covers bad backend names, missing required fields, invalid field type
    strings, and other configuration issues caught before any network
    requests are made.

    Examples:
        - Backend set to "unknown" instead of "http" or "browser"
        - Field type set to "dict" which is not supported
        - No API keys provided and none found in environment
    """

    pass


class AllKeysExhaustedError(ScraperError):
    """Raised when all configured API keys have been tried and failed.

    This happens during key rotation: if the primary key gets a 429 or
    auth error, the library rotates to backup keys. If all keys fail,
    this exception is raised.

    The ``details`` dict will contain:
        - ``attempted_keys``: number of keys that were tried
        - ``last_error``: the error message from the last attempted key
    """

    def __init__(
        self,
        message: str = "All API keys have been exhausted",
        attempted_keys: int = 0,
        last_error: str | None = None,
        details: dict | None = None,
    ) -> None:
        combined = dict(details or {})
        combined["attempted_keys"] = attempted_keys
        if last_error is not None:
            combined["last_error"] = last_error
        super().__init__(message, combined)
        self.attempted_keys = attempted_keys
        self.last_error = last_error
