"""Configuration management for the scraper library.

This module provides:

- **load_config()**: Factory function that creates a ScraperConfig with
  environment-based defaults and user overrides.
- **KeyRotator**: A small stateful class that cycles through API keys
  when rate-limit or authentication errors occur.

API keys are resolved in this order:
1. Explicitly passed in ``api_keys`` list
2. ``OPENAI_API_KEY`` environment variable (primary)
3. ``OPENAI_API_KEY_BACKUP_1``, ``OPENAI_API_KEY_BACKUP_2``, ... (backups)

If no keys are found anywhere, operations that require an LLM will raise
``ConfigError`` at call time rather than at config creation time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from scraper.exceptions import AllKeysExhaustedError, ConfigError
from scraper.models import ScraperConfig

__all__ = [
    "load_config",
    "KeyRotator",
]

logger = logging.getLogger(__name__)


def load_config(**overrides: Any) -> ScraperConfig:
    """Create a ScraperConfig with sensible defaults and optional overrides.

    Any keyword argument matching a ScraperConfig field name will override
    the default value. API keys are loaded from environment variables if
    not provided.

    Args:
        **overrides: Keyword arguments matching ScraperConfig field names.
                     For example: ``load_config(backend="browser", max_pages=100)``.

    Returns:
        A fully initialized ScraperConfig.

    Raises:
        ConfigError: If an override key does not match any config field.

    Examples:
        >>> config = load_config(backend="browser", max_concurrent=3)
        >>> config.backend
        'browser'
    """
    valid_fields = set(ScraperConfig.model_fields.keys())
    invalid = set(overrides.keys()) - valid_fields
    if invalid:
        raise ConfigError(
            f"Unknown config fields: {sorted(invalid)}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    return ScraperConfig(**overrides)


class KeyRotator:
    """Manages API key rotation for handling rate-limit and auth failures.

    The rotator maintains a list of keys and a current index. On failure,
    call ``rotate()`` to move to the next key. When all keys have been
    tried, ``AllKeysExhaustedError`` is raised.

    Attributes:
        keys: List of API key strings.
        current_index: Index of the currently active key.

    Examples:
        >>> rotator = KeyRotator(["key1", "key2", "key3"])
        >>> rotator.current_key
        'key1'
        >>> rotator.rotate("rate limited")
        'key2'
    """

    def __init__(self, keys: List[str]) -> None:
        if not keys:
            raise ConfigError(
                "No API keys available. Set OPENAI_API_KEY environment variable "
                "or pass api_keys to ScraperConfig."
            )
        self._keys = list(keys)
        self._current_index = 0
        self._attempts = 0
        self._max_attempts = len(self._keys)

    @property
    def current_key(self) -> str:
        """Return the currently active API key."""
        return self._keys[self._current_index]

    @property
    def total_keys(self) -> int:
        """Return the total number of available keys."""
        return len(self._keys)

    def rotate(self, reason: str = "") -> str:
        """Rotate to the next API key.

        Args:
            reason: Human-readable reason for rotation (e.g., "429 rate limit").

        Returns:
            The next API key to use.

        Raises:
            AllKeysExhaustedError: If all keys have been attempted.
        """
        self._attempts += 1
        if self._attempts >= self._max_attempts:
            raise AllKeysExhaustedError(
                message=f"All {self._max_attempts} API keys exhausted",
                attempted_keys=self._max_attempts,
                last_error=reason,
            )

        self._current_index = (self._current_index + 1) % len(self._keys)
        logger.warning(
            "Rotating API key (attempt %d/%d, reason: %s)",
            self._attempts + 1,
            self._max_attempts,
            reason or "unknown",
        )
        return self._keys[self._current_index]

    def reset(self) -> None:
        """Reset the attempt counter (e.g., after a successful request).

        Call this after a successful LLM call to allow future rotations
        to start fresh.
        """
        self._attempts = 0
