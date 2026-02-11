# Architecture Document

## Overview

This library is a pip-installable Python package (`scraper`) that provides LLM-based data extraction from web pages. It wraps crawl4ai for fetching and uses OpenAI-compatible LLMs (via litellm) for structured extraction.

The library supports three modes of operation (single-URL, multi-URL, crawl) and two fetching backends (HTTP, browser). All modes share the same field definition input and output models.

---

## Package Structure

```
scraper/
  __init__.py          Public API exports (__all__ controlled)
  models.py            Pydantic data models (FieldDefinition, ScrapeResult, CrawlState, ScraperConfig)
  config.py            Config factory (load_config), API key rotation (KeyRotator)
  extraction.py        SHARED extraction logic (schema gen, instructions, parsing, merge, completion)
  http_backend.py      HTTP backend stub (AsyncHTTPCrawlerStrategy)
  browser_backend.py   Browser backend stub (Playwright via AsyncWebCrawler)
  api.py               Scraper class, define_fields helper, sync wrappers
  cli.py               CLI entry point (argparse)
  exceptions.py        Exception hierarchy
```

---

## Module Responsibilities

### models.py
Defines all data structures. No business logic. No imports from other scraper modules.

| Model | Purpose | Frozen? |
|-------|---------|---------|
| `FieldDefinition` | User input: what to extract | Yes |
| `ScrapeResult` | Output: extraction result per URL | Yes |
| `CrawlState` | Internal: crawl session tracking | No (mutated in-place) |
| `ScraperConfig` | Configuration: backend, LLM, concurrency | No |

### config.py
Depends on: `models.py`, `exceptions.py`

- `load_config(**overrides)` -- factory that validates overrides and returns ScraperConfig
- `KeyRotator` -- stateful class that cycles through API keys on 429/auth failures

### extraction.py
Depends on: `models.py`, `config.py`, `exceptions.py`, `crawl4ai` (lazy imports)

This is the HEART of the library. Contains all shared logic that both backends use:
- Schema and instruction generation from field definitions
- crawl4ai strategy/config creation (the only place LLMExtractionStrategy is instantiated)
- Result parsing (JSON string from crawl4ai to validated dict)
- Completion detection (all scalar fields filled?)
- Result merging (scalar: first non-null; list: union + dedup)
- URL normalization and domain extraction
- URL prioritization for crawl mode

### http_backend.py
Depends on: `models.py`, `config.py`, `extraction.py`, `exceptions.py`, `crawl4ai`

Implements scrape_one, scrape_many, crawl using AsyncHTTPCrawlerStrategy.

### browser_backend.py
Depends on: `models.py`, `config.py`, `extraction.py`, `exceptions.py`, `crawl4ai`

Implements scrape_one, scrape_many, crawl using AsyncWebCrawler with BrowserConfig.

### api.py
Depends on: `models.py`, `config.py`, `http_backend.py`, `browser_backend.py`, `exceptions.py`

Public-facing Scraper class and helpers. Routes to appropriate backend.

### cli.py
Depends on: `api.py`, `models.py`

Thin CLI wrapper using argparse.

### exceptions.py
No internal dependencies. All other modules import from it.

---

## Dependency Graph

```
exceptions.py  (no deps)
     ^
     |
models.py  (no internal deps)
     ^
     |
config.py  (depends on: models, exceptions)
     ^
     |
extraction.py  (depends on: models, config, exceptions)
     ^         (lazy-imports crawl4ai for strategy creation)
     |
     +------ http_backend.py  (depends on: models, config, extraction, exceptions)
     |
     +------ browser_backend.py  (depends on: models, config, extraction, exceptions)
     |
     v
api.py  (depends on: models, config, http_backend, browser_backend, exceptions)
     ^
     |
cli.py  (depends on: api, models)
     ^
     |
__init__.py  (re-exports from api, models, config, exceptions)
```

---

## Data Flow

### Single-URL Mode

```
User
  |
  v
Scraper.scrape(url, fields)
  |
  v
Backend.scrape_one(url, fields)
  |
  +---> extraction.create_extraction_strategy(fields, config, api_key)
  |       |
  |       +---> extraction.build_json_schema(fields)
  |       +---> extraction.build_instruction(fields)
  |       +---> LLMExtractionStrategy(llm_config, instruction, schema, ...)
  |
  +---> extraction.create_run_config(strategy, stream=False)
  |       |
  |       +---> CrawlerRunConfig(extraction_strategy, cache_mode=BYPASS)
  |
  +---> crawler.arun(url, config=run_config)
  |       |
  |       +---> [crawl4ai fetches page, runs LLM extraction]
  |       +---> returns CrawlResult with extracted_content (JSON string)
  |
  +---> extraction.parse_extraction_result(result.extracted_content, fields)
  |       |
  |       +---> json.loads(), unwrap to dict, normalize values
  |
  +---> ScrapeResult(url=url, success=True, data=parsed_data)
  |
  v
User receives ScrapeResult
```

### Multi-URL Mode

```
User
  |
  v
Scraper.scrape_many(urls, fields)  -- async generator
  |
  v
Backend.scrape_many(urls, fields)
  |
  +---> Deduplicate URLs via normalize_url()
  +---> Create shared extraction strategy
  +---> Create run config with stream=True, semaphore_count=max_concurrent
  +---> crawler.arun_many(urls, config)  -- returns async generator
  |
  +---> for each result in stream:
  |       +---> parse_extraction_result()
  |       +---> yield ScrapeResult
  |
  v
User receives ScrapeResults as they complete (async for)
```

### Crawl Mode

```
User
  |
  v
Scraper.crawl(start_url, fields, max_pages)  -- async generator
  |
  v
Backend.crawl(start_url, fields, max_pages)
  |
  +---> Extract domain from start_url
  +---> Create BFSDeepCrawlStrategy(max_depth, max_pages, filter_chain)
  +---> Create extraction strategy
  +---> Create run config with deep_crawl_strategy + stream=True
  +---> Initialize CrawlState
  |
  +---> crawler.arun(start_url, config)  -- returns async generator (deep crawl + stream)
  |
  +---> for each result in stream:
  |       +---> parse_extraction_result()
  |       +---> update_crawl_state(state, new_data, fields, url)
  |       |       +---> merge_results(existing, new, fields)
  |       |       +---> is_complete(merged_data, fields)
  |       +---> yield ScrapeResult
  |       +---> if state.is_complete and not has_list_fields:
  |               +---> await bfs_strategy.shutdown()  -- early stop
  |               +---> break
  |
  v
User receives ScrapeResults incrementally (async for)
```

---

## Completion Detection Algorithm (Crawl Mode)

The completion detection determines when to stop crawling early (before max_pages is reached).

### Rules

1. **Check for list fields**: If ANY FieldDefinition has a type starting with `"list["`, the data can NEVER be considered complete because we do not know the total count of list items. In this case, crawling continues until max_pages or no more links.

2. **Check scalar fields**: For each FieldDefinition with a scalar type (string, number, boolean, integer), check if `merged_data[field.name]` is not None.

3. **Complete if**: No list fields exist AND all scalar fields have non-null values in merged_data.

### Pseudocode

```python
def is_complete(merged_data, fields):
    if any(field.is_list for field in fields):
        return False
    return all(
        merged_data.get(field.name) is not None
        for field in fields
        if field.is_scalar
    )
```

### When completion is detected

The backend calls `await bfs_strategy.shutdown()` which sets an internal cancel event in crawl4ai's BFSDeepCrawlStrategy, causing it to stop discovering new links and finish the current batch.

---

## Result Merging Algorithm

When crawling multiple pages, extraction results are merged according to field type:

### Scalar Fields (string, number, boolean, integer)
- **Rule**: Keep the first non-null value found.
- **Rationale**: Scalar fields have one correct answer. The first page that provides it is likely the most relevant.

### List Fields (list[string], list[number], list[object], list[boolean])
- **Rule**: Union the lists from all pages and deduplicate.
- **Dedup for primitives**: Direct equality comparison, preserving insertion order.
- **Dedup for objects**: JSON serialization with sorted keys for comparison.
- **Rationale**: Lists accumulate across pages (e.g., team members on /about vs /team).

---

## Resource Lifecycle

### HTTP Backend

```
__aenter__():
  1. Create HTTPCrawlerConfig()
  2. Create AsyncHTTPCrawlerStrategy(browser_config=http_config)
  3. Create AsyncWebCrawler(crawler_strategy=http_strategy)
  4. await crawler.start()  --> creates aiohttp.ClientSession
  5. Create KeyRotator from config.api_keys

[scraping calls reuse the same session]

__aexit__():
  1. await crawler.close()  --> closes aiohttp.ClientSession
  2. Set crawler = None
```

### Browser Backend

```
__aenter__():
  1. Create BrowserConfig(headless=True, enable_stealth=True, text_mode=True)
  2. Create AsyncWebCrawler(config=browser_config)
  3. await crawler.start()  --> launches Playwright + Chromium
  4. Create KeyRotator from config.api_keys

[scraping calls reuse the same browser, each URL gets a new tab that is closed after extraction]

__aexit__():
  1. await crawler.close()  --> shuts down Chromium process
  2. Set crawler = None
```

### Safety Net

The Scraper class enforces context manager usage:
- `_ensure_entered()` raises RuntimeError if used without `async with`
- All backend operations go through this check

---

## API Key Rotation

```
KeyRotator(keys=["key1", "key2", "key3"])
  |
  +-- current_key = "key1" (index 0, attempts 0)
  |
  +-- On 429/auth error: rotate("rate limited")
  |     +-- attempts = 1
  |     +-- index = 1
  |     +-- returns "key2"
  |
  +-- On success: reset()
  |     +-- attempts = 0
  |
  +-- On all keys fail: AllKeysExhaustedError
        +-- attempts >= len(keys)
```

The extraction module creates a new `LLMExtractionStrategy` with the rotated key. Each backend is responsible for catching LLM errors and triggering rotation.

---

## Backend Interface Contract

Both HTTPBackend and BrowserBackend implement the same three methods:

```python
async def scrape_one(url, fields) -> ScrapeResult
async def scrape_many(urls, fields) -> AsyncGenerator[ScrapeResult, None]
async def crawl(start_url, fields, max_pages, max_depth) -> AsyncGenerator[ScrapeResult, None]
```

Both are async context managers (`__aenter__` / `__aexit__`).

Both import all extraction logic from `scraper.extraction` -- they do NOT duplicate schema generation, instruction building, result parsing, or completion detection.

---

## crawl4ai Integration Points

The library uses crawl4ai at these specific points:

| Our function | crawl4ai class | crawl4ai module |
|-------------|----------------|-----------------|
| `create_extraction_strategy()` | `LLMExtractionStrategy`, `LLMConfig` | `crawl4ai` |
| `create_run_config()` | `CrawlerRunConfig`, `CacheMode` | `crawl4ai` |
| `HTTPBackend.__aenter__()` | `AsyncHTTPCrawlerStrategy`, `HTTPCrawlerConfig`, `AsyncWebCrawler` | `crawl4ai`, `crawl4ai.async_crawler_strategy` |
| `BrowserBackend.__aenter__()` | `AsyncWebCrawler`, `BrowserConfig` | `crawl4ai` |
| `Backend.crawl()` | `BFSDeepCrawlStrategy`, `FilterChain`, `DomainFilter` | `crawl4ai` |
| Both backends | `AsyncWebCrawler.arun()`, `.arun_many()` | `crawl4ai` |

All crawl4ai imports in extraction.py are LAZY (inside function bodies) to avoid import errors when crawl4ai is not installed.

---

## Import Path Reference (from CRAWL4AI_RESEARCH.md)

```python
# Core
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# HTTP-only (NOT in crawl4ai.__init__)
from crawl4ai import HTTPCrawlerConfig
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

# LLM extraction
from crawl4ai import LLMExtractionStrategy, LLMConfig

# Deep crawling
from crawl4ai import BFSDeepCrawlStrategy, FilterChain, DomainFilter, ContentTypeFilter

# Dispatchers (for arun_many)
from crawl4ai import SemaphoreDispatcher
```

---

## Configuration Defaults

| Setting | Default | Rationale |
|---------|---------|-----------|
| backend | "http" | Faster, no browser overhead for static pages |
| model | "openai/gpt-4o-mini" | Good balance of cost and quality |
| max_concurrent | 5 | Safe default for HTTP; users should set 3 for browser |
| timeout | 30000ms | Generous but bounded |
| headless | True | No visible browser window |
| stealth | True | Avoid bot detection by default |
| max_pages | 50 | Reasonable cap for crawl mode |
| max_depth | 3 | Prevents going too deep |
| input_format | "markdown" | Best token efficiency per the research |
| chunk_token_threshold | 2048 | crawl4ai default |
| verbose | False | Quiet by default |

---

## Field Type System

| Type | JSON Schema | Category | Completion? |
|------|-------------|----------|-------------|
| string | `{"type": "string"}` | Scalar | Yes -- complete when non-null |
| number | `{"type": "number"}` | Scalar | Yes |
| integer | `{"type": "integer"}` | Scalar | Yes |
| boolean | `{"type": "boolean"}` | Scalar | Yes |
| list[string] | `{"type": "array", "items": {"type": "string"}}` | List | Never complete |
| list[number] | `{"type": "array", "items": {"type": "number"}}` | List | Never complete |
| list[boolean] | `{"type": "array", "items": {"type": "boolean"}}` | List | Never complete |
| list[object] | `{"type": "array", "items": {"type": "object"}}` | List | Never complete |

---

## Exception Hierarchy

```
ScraperError (base)
  +-- FetchError          -- network/HTTP failures
  +-- ExtractionError     -- LLM/parsing failures
  +-- ConfigError         -- bad configuration
  +-- AllKeysExhaustedError -- all API keys failed
```

Each exception carries a `message` string and a `details` dict with additional context (URL, status code, raw response, etc.).

---

## Design Decisions

### 1. Why flat package layout (not src/)?
Simpler for a single-package project. The `scraper/` directory IS the package. pyproject.toml's `[tool.hatch.build.targets.wheel] packages = ["scraper"]` handles this.

### 2. Why frozen models for FieldDefinition and ScrapeResult?
These are value objects. Once created, they should not be mutated. This prevents accidental modification and enables safe sharing across async tasks.

### 3. Why is CrawlState NOT frozen?
CrawlState is mutated in-place during crawl sessions for efficiency. Creating new instances on every page would be wasteful for large crawls.

### 4. Why lazy imports for crawl4ai in extraction.py?
The extraction module contains pure logic functions (merge, complete, normalize) that should work without crawl4ai installed. Only the strategy/config creation functions need crawl4ai, so those import it inside the function body.

### 5. Why a KeyRotator class instead of simple index tracking?
Encapsulates rotation logic, attempt counting, and the "all keys exhausted" error in one place. Backends hold a reference and call rotate() on failure, reset() on success.

### 6. Why does crawl mode use BFSDeepCrawlStrategy?
crawl4ai already implements BFS link discovery, URL dedup, depth tracking, and streaming. Reimplementing this would be error-prone. We add completion detection on top via strategy.shutdown().

### 7. Why input_format defaults to "markdown"?
Per the research: "Good balance of quality and token efficiency." fit_markdown requires a content filter setup; HTML uses more tokens. Markdown is the safest default.

### 8. Why CacheMode.BYPASS?
We always want fresh content. The library is for real-time extraction, not cached results.
