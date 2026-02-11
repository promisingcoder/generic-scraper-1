# crawl4ai Library Research Document

**Version investigated:** 0.8.0
**Python version:** 3.14.0
**Platform:** Windows 11
**Date:** 2026-02-11
**Package location:** `C:\Users\Yossef\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\crawl4ai\`

All findings below are verified by reading actual source code and running import tests.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [AsyncWebCrawler and BrowserConfig (Browser Mode)](#2-asyncwebcrawler-and-browserconfig-browser-mode)
3. [AsyncHTTPCrawlerStrategy and HTTPCrawlerConfig (HTTP-Only Mode)](#3-asynchttpcrawlerstrategy-and-httpcrawlerconfig-http-only-mode)
4. [Switching Between HTTP and Browser Modes](#4-switching-between-http-and-browser-modes)
5. [LLMExtractionStrategy and LLMConfig](#5-llmextractionstrategy-and-llmconfig)
6. [CrawlResult](#6-crawlresult)
7. [CrawlerRunConfig](#7-crawlerrunconfig)
8. [CacheMode](#8-cachemode)
9. [arun() and arun_many()](#9-arun-and-arun_many)
10. [Deep Crawling](#10-deep-crawling)
11. [Content Pipeline and Markdown Generation](#11-content-pipeline-and-markdown-generation)
12. [Resource Lifecycle](#12-resource-lifecycle)
13. [Verified Import Paths](#13-verified-import-paths)
14. [Gotchas and Version-Specific Issues](#14-gotchas-and-version-specific-issues)
15. [Recommendations for the Scraper Project](#15-recommendations-for-the-scraper-project)

---

## 1. Architecture Overview

```
AsyncWebCrawler
  |
  |-- config: BrowserConfig (or passed via constructor)
  |-- crawler_strategy: AsyncCrawlerStrategy (abstract)
  |     |-- AsyncPlaywrightCrawlerStrategy (default, browser-based)
  |     |-- AsyncHTTPCrawlerStrategy (HTTP-only, no browser)
  |
  |-- arun(url, config: CrawlerRunConfig) -> CrawlResult
  |-- arun_many(urls, config, dispatcher) -> List[CrawlResult] | AsyncGenerator
  |
  |-- DeepCrawlDecorator wraps arun() to intercept deep_crawl_strategy
```

**Key architectural decision:** The crawl strategy is set at `AsyncWebCrawler` construction time. You cannot switch between HTTP-only and browser mode on a per-request basis. You need separate crawler instances for each mode.

**Source files:**
- `async_webcrawler.py` -- main crawler class
- `async_configs.py` -- all config classes (BrowserConfig, HTTPCrawlerConfig, CrawlerRunConfig, LLMConfig)
- `async_crawler_strategy.py` -- AsyncPlaywrightCrawlerStrategy, AsyncHTTPCrawlerStrategy
- `extraction_strategy.py` -- LLMExtractionStrategy, ExtractionStrategy base
- `models.py` -- CrawlResult, MarkdownGenerationResult, Link, Links
- `cache_context.py` -- CacheMode enum
- `deep_crawling/` -- BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, filters, scorers

---

## 2. AsyncWebCrawler and BrowserConfig (Browser Mode)

### Import Path
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig
```

### Constructor: AsyncWebCrawler
```python
class AsyncWebCrawler:
    def __init__(
        self,
        crawler_strategy: AsyncCrawlerStrategy = None,  # defaults to AsyncPlaywrightCrawlerStrategy
        config: BrowserConfig = None,                    # defaults to BrowserConfig()
        base_directory: str = str(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home())),
        thread_safe: bool = False,
        logger: AsyncLoggerBase = None,
        **kwargs,
    )
```

If `crawler_strategy` is None (the default), it creates `AsyncPlaywrightCrawlerStrategy(browser_config=config)`.

### Constructor: BrowserConfig
```python
class BrowserConfig:
    def __init__(
        self,
        browser_type: str = "chromium",          # "chromium", "firefox", "webkit"
        headless: bool = True,
        browser_mode: str = "dedicated",         # "builtin", "dedicated", "custom", "docker"
        use_managed_browser: bool = False,
        cdp_url: str = None,
        browser_context_id: str = None,
        target_id: str = None,
        cdp_cleanup_on_close: bool = False,
        create_isolated_context: bool = False,
        use_persistent_context: bool = False,
        user_data_dir: str = None,
        chrome_channel: str = "chromium",
        channel: str = "chromium",
        proxy: str = None,                       # DEPRECATED - use proxy_config
        proxy_config: Union[ProxyConfig, dict, None] = None,
        viewport_width: int = 1080,
        viewport_height: int = 600,
        viewport: dict = None,
        accept_downloads: bool = False,
        downloads_path: str = None,
        storage_state: Union[str, dict, None] = None,
        ignore_https_errors: bool = True,
        java_script_enabled: bool = True,
        sleep_on_close: bool = False,
        verbose: bool = True,
        cookies: list = None,
        headers: dict = None,
        user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36",
        user_agent_mode: str = "",               # "random" for random UA
        user_agent_generator_config: dict = {},
        text_mode: bool = False,                 # disables images for faster loads
        light_mode: bool = False,
        extra_args: list = None,
        debugging_port: int = 9222,
        host: str = "localhost",
        enable_stealth: bool = False,            # playwright-stealth for anti-bot
        init_scripts: List[str] = None,
    )
```

### Browser Lifecycle

1. Browser is created when `start()` is called (or when entering `async with`)
2. It stays alive for the lifetime of the crawler instance
3. Browser is destroyed when `close()` is called (or when exiting `async with`)
4. If you call `arun()` without `start()`, it auto-starts (line 241 in async_webcrawler.py)

### Usage Pattern (Browser Mode)
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

browser_config = BrowserConfig(
    headless=True,
    browser_type="chromium",
    verbose=False,
)

async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun(
        url="https://example.com",
        config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
    )
    print(result.markdown)
```

---

## 3. AsyncHTTPCrawlerStrategy and HTTPCrawlerConfig (HTTP-Only Mode)

### Import Path
```python
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy
from crawl4ai import HTTPCrawlerConfig
```

**IMPORTANT:** `AsyncHTTPCrawlerStrategy` is NOT exported from `crawl4ai.__init__`. You must import it from `crawl4ai.async_crawler_strategy`.

### Constructor: HTTPCrawlerConfig
```python
class HTTPCrawlerConfig:
    def __init__(
        self,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
    )
```

### Constructor: AsyncHTTPCrawlerStrategy
```python
class AsyncHTTPCrawlerStrategy(AsyncCrawlerStrategy):
    DEFAULT_TIMEOUT: Final[int] = 30
    DEFAULT_CHUNK_SIZE: Final[int] = 64 * 1024
    DEFAULT_MAX_CONNECTIONS: Final[int] = min(32, (os.cpu_count() or 1) * 4)
    DEFAULT_DNS_CACHE_TTL: Final[int] = 300

    def __init__(
        self,
        browser_config: Optional[HTTPCrawlerConfig] = None,  # NOTE: parameter named "browser_config" but takes HTTPCrawlerConfig
        logger: Optional[AsyncLogger] = None,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        dns_cache_ttl: int = DEFAULT_DNS_CACHE_TTL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    )
```

**Session management:** Uses `aiohttp.ClientSession` internally. Session is created on `start()` and closed on `close()`.

**Headers:** Has built-in base headers:
```python
_BASE_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
```

HTTPCrawlerConfig headers are merged on top of these.

### Usage Pattern (HTTP-Only Mode)
```python
from crawl4ai import AsyncWebCrawler, HTTPCrawlerConfig, CrawlerRunConfig, CacheMode
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

http_config = HTTPCrawlerConfig(
    method="GET",
    headers={"Authorization": "Bearer token123"},
    follow_redirects=True,
)

http_strategy = AsyncHTTPCrawlerStrategy(browser_config=http_config)

async with AsyncWebCrawler(crawler_strategy=http_strategy) as crawler:
    result = await crawler.arun(
        url="https://example.com",
        config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
    )
    print(result.markdown)
```

---

## 4. Switching Between HTTP and Browser Modes

**You CANNOT switch modes per-request.** The strategy is fixed at crawler construction time.

To support both modes, you need two separate crawler instances:

```python
# HTTP-only crawler
http_strategy = AsyncHTTPCrawlerStrategy(browser_config=HTTPCrawlerConfig())
http_crawler = AsyncWebCrawler(crawler_strategy=http_strategy)

# Browser-based crawler
browser_crawler = AsyncWebCrawler(config=BrowserConfig(headless=True))

# Use the appropriate one per URL
await http_crawler.start()
await browser_crawler.start()

result_http = await http_crawler.arun(url="https://static-site.com", config=run_config)
result_browser = await browser_crawler.arun(url="https://js-heavy-site.com", config=run_config)

await http_crawler.close()
await browser_crawler.close()
```

---

## 5. LLMExtractionStrategy and LLMConfig

### Import Paths
```python
from crawl4ai import LLMExtractionStrategy, LLMConfig
```

### Constructor: LLMConfig
```python
class LLMConfig:
    def __init__(
        self,
        provider: str = "openai/gpt-4o",         # format: "<provider>/<model>"
        api_token: Optional[str] = None,           # explicit token, or "env:VAR_NAME"
        base_url: Optional[str] = None,            # custom API endpoint
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        n: Optional[int] = None,
        backoff_base_delay: Optional[int] = None,     # default: 2
        backoff_max_attempts: Optional[int] = None,    # default: 3
        backoff_exponential_factor: Optional[int] = None,  # default: 2
    )
```

### API Key Resolution (from source, lines 1901-1917 in async_configs.py)

The resolution order for `api_token` in `LLMConfig.__init__`:
1. If `api_token` is provided and does NOT start with "env:": use it directly
2. If `api_token` starts with "env:": read from `os.getenv(api_token[4:])`
3. If `api_token` is None: check if provider matches a known prefix in `PROVIDER_MODELS_PREFIXES`
4. If no match: falls back to `os.getenv("OPENAI_API_KEY")`

**Known provider prefixes and their env vars** (from config.py):
```python
PROVIDER_MODELS_PREFIXES = {
    "ollama": "no-token-needed",
    "groq": os.getenv("GROQ_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}
```

**For our project, the best approach is to always pass `api_token` explicitly:**
```python
llm_config = LLMConfig(
    provider="openai/gpt-4o-mini",
    api_token="sk-...",  # or "env:OPENAI_API_KEY" to read from env
)
```

### Constructor: LLMExtractionStrategy
```python
class LLMExtractionStrategy(ExtractionStrategy):
    def __init__(
        self,
        llm_config: LLMConfig = None,                 # PREFERRED way to pass LLM config
        instruction: str = None,                        # user instruction for extraction
        schema: Dict = None,                            # Pydantic model schema dict or JSON schema
        extraction_type: str = "block",                 # "block" or "schema"
        chunk_token_threshold: int = 2048,              # max tokens per chunk
        overlap_rate: float = 0.1,
        word_token_rate: float = 1.3,
        apply_chunking: bool = True,
        input_format: str = "markdown",                 # "markdown", "html", "fit_markdown", "cleaned_html", "fit_html"
        force_json_response: bool = False,
        verbose: bool = False,
        # DEPRECATED - use llm_config instead:
        provider: str = "openai/gpt-4o",
        api_token: Optional[str] = None,
        base_url: str = None,
        api_base: str = None,
        **kwargs,  # extra_args for API request params
    )
```

**CRITICAL:** Setting the deprecated `provider`, `api_token`, `base_url`, or `api_base` directly on LLMExtractionStrategy will raise `AttributeError` (enforced by `__setattr__` override). Always use `llm_config=LLMConfig(...)`.

### How extraction_type Works

- **`extraction_type="block"`** (default): Uses `PROMPT_EXTRACT_BLOCKS` or `PROMPT_EXTRACT_BLOCKS_WITH_INSTRUCTION` prompts. Returns the content broken into semantically meaningful blocks with index, tags, and content.
- **`extraction_type="schema"`**: Uses `PROMPT_EXTRACT_SCHEMA_WITH_INSTRUCTION`. Requires a `schema` dict. Returns structured data matching the schema.
- If `schema` is provided, `extraction_type` is **automatically set to "schema"** (line 571).
- If `extraction_type="schema"` but no `schema` is provided: uses `PROMPT_EXTRACT_INFERRED_SCHEMA` to auto-infer a schema.

### How to Pass a Pydantic Model

Pass the `.model_json_schema()` output:

```python
from pydantic import BaseModel
from typing import List

class Lawyer(BaseModel):
    name: str
    title: str
    email: str = ""

class LawyerList(BaseModel):
    lawyers: List[Lawyer]

strategy = LLMExtractionStrategy(
    llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token="sk-..."),
    instruction="Extract all lawyers listed on this page.",
    schema=LawyerList.model_json_schema(),
    # extraction_type is auto-set to "schema" because schema is provided
    input_format="markdown",
)
```

### How input_format Works

Determines which content representation is sent to the LLM (from async_webcrawler.py line 699):

```python
content = {
    "markdown": markdown_result.raw_markdown,
    "html": html,                                # original fetched HTML
    "fit_html": fit_html,                        # filtered HTML
    "cleaned_html": cleaned_html,                # cleaned HTML
    "fit_markdown": markdown_result.fit_markdown, # requires content_filter on markdown_generator
}.get(content_format, markdown_result.raw_markdown)
```

- **"markdown"** (default): Raw markdown generated from cleaned HTML. Good balance of quality and token efficiency.
- **"html"**: Raw HTML. Useful when CSS selectors or structure matter. Uses `IdentityChunking` (no chunking).
- **"fit_markdown"**: Filtered markdown. Requires a content filter on the `DefaultMarkdownGenerator`. Falls back to "markdown" if not available.
- **"cleaned_html"**: Cleaned HTML. Uses `IdentityChunking`.
- **"fit_html"**: Filtered HTML. Uses `IdentityChunking`.

### Chunking Behavior

When `input_format` is "html", "cleaned_html", or "fit_html", the chunking strategy is replaced with `IdentityChunking()` (no splitting). For markdown formats, the `config.chunking_strategy` is used (default: `RegexChunking()`).

### How the LLM Prompt is Constructed

The prompt templates are in `prompts.py`. Variables are substituted using simple string replacement:
- `{URL}` -- the page URL
- `{HTML}` -- the content (despite the name, this is whatever `input_format` selects)
- `{REQUEST}` -- the `instruction` parameter
- `{SCHEMA}` -- `json.dumps(schema, indent=2)`

The content is sanitized via `escape_json_string(sanitize_html(content))`.

### API Call Mechanism

Uses `perform_completion_with_backoff()` from `utils.py` which calls **litellm** under the hood. The provider string format (e.g., "openai/gpt-4o-mini") is the litellm format.

### Token Usage Tracking

```python
strategy = LLMExtractionStrategy(...)
# After extraction:
print(strategy.total_usage)  # TokenUsage(completion_tokens=..., prompt_tokens=..., total_tokens=...)
print(strategy.usages)       # List of per-chunk TokenUsage objects
```

### Complete Usage Example
```python
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig,
    LLMExtractionStrategy, LLMConfig, CacheMode,
)

llm_config = LLMConfig(
    provider="openai/gpt-4o-mini",
    api_token="sk-your-key-here",
)

strategy = LLMExtractionStrategy(
    llm_config=llm_config,
    instruction="Extract all product names and prices.",
    schema={"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "price": {"type": "string"}}}},
    input_format="markdown",
    chunk_token_threshold=4096,
    verbose=True,
)

run_config = CrawlerRunConfig(
    extraction_strategy=strategy,
    cache_mode=CacheMode.BYPASS,
)

async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
    result = await crawler.arun(url="https://example.com/products", config=run_config)
    if result.success:
        import json
        data = json.loads(result.extracted_content)
        print(data)
```

---

## 6. CrawlResult

### Import Path
```python
from crawl4ai import CrawlResult
```

### All Fields (verified via model_fields inspection)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `url` | `str` | required | The crawled URL |
| `html` | `str` | required | Raw HTML content |
| `success` | `bool` | required | Whether crawl succeeded |
| `cleaned_html` | `Optional[str]` | None | HTML after cleaning |
| `media` | `Dict[str, List[Dict]]` | {} | Images, videos, audios |
| `links` | `Dict[str, List[Dict]]` | {} | `{"internal": [...], "external": [...]}` |
| `downloaded_files` | `Optional[List[str]]` | None | Downloaded file paths |
| `js_execution_result` | `Optional[Dict]` | None | JS execution results |
| `screenshot` | `Optional[str]` | None | Base64 screenshot |
| `pdf` | `Optional[bytes]` | None | PDF data |
| `mhtml` | `Optional[str]` | None | MHTML data |
| `extracted_content` | `Optional[str]` | None | JSON string from extraction strategy |
| `metadata` | `Optional[dict]` | None | Page metadata |
| `error_message` | `Optional[str]` | None | Error details if failed |
| `session_id` | `Optional[str]` | None | Session identifier |
| `response_headers` | `Optional[dict]` | None | HTTP response headers |
| `status_code` | `Optional[int]` | None | HTTP status code |
| `ssl_certificate` | `Optional[SSLCertificate]` | None | SSL cert info |
| `dispatch_result` | `Optional[DispatchResult]` | None | Dispatch metadata from arun_many |
| `redirected_url` | `Optional[str]` | None | Final URL after redirects |
| `network_requests` | `Optional[List[Dict]]` | None | Captured network requests |
| `console_messages` | `Optional[List[Dict]]` | None | Browser console messages |
| `tables` | `List[Dict]` | [] | Extracted table data |
| `head_fingerprint` | `Optional[str]` | None | Cache fingerprint |
| `cached_at` | `Optional[float]` | None | Cache timestamp |
| `cache_status` | `Optional[str]` | None | "hit", "hit_validated", "hit_fallback", "miss" |

### The `markdown` Property

`markdown` is NOT a regular field -- it is a private attribute `_markdown` of type `MarkdownGenerationResult`, exposed via a property that returns a `StringCompatibleMarkdown` object.

`StringCompatibleMarkdown` extends `str` so it acts like a string (the raw markdown), but also gives access to:
- `result.markdown.raw_markdown` -- raw markdown string
- `result.markdown.markdown_with_citations` -- markdown with citation links
- `result.markdown.references_markdown` -- reference section
- `result.markdown.fit_markdown` -- filtered markdown (requires content filter)
- `result.markdown.fit_html` -- filtered HTML

**Using as string:** `str(result.markdown)` or `print(result.markdown)` returns the raw markdown.

### The `links` Field Structure

`links` is `Dict[str, List[Dict]]` with keys `"internal"` and `"external"`. Each dict in the list has:
- `href`: URL string
- `text`: Link text
- `title`: Link title attribute
- `base_domain`: Domain of the link

```python
# Accessing links
internal_links = result.links.get("internal", [])
external_links = result.links.get("external", [])
for link in internal_links:
    print(link["href"], link.get("text", ""))
```

### The `extracted_content` Field

When an extraction strategy is used, `extracted_content` is a **JSON string** (not a dict). You must `json.loads()` it:

```python
import json
if result.extracted_content:
    data = json.loads(result.extracted_content)
```

---

## 7. CrawlerRunConfig

### Import Path
```python
from crawl4ai import CrawlerRunConfig
```

### Key Parameters (there are many -- only the most important listed)

**Content Processing:**
- `extraction_strategy: ExtractionStrategy = None` -- LLMExtractionStrategy, JsonCssExtractionStrategy, etc.
- `chunking_strategy: ChunkingStrategy = RegexChunking()`
- `markdown_generator: MarkdownGenerationStrategy = DefaultMarkdownGenerator()`
- `word_count_threshold: int = 1` (MIN_WORD_THRESHOLD from config)
- `css_selector: str = None` -- CSS selector to extract a specific portion
- `target_elements: List[str] = None` -- CSS selectors for specific elements
- `only_text: bool = False`

**Caching:**
- `cache_mode: CacheMode = CacheMode.BYPASS` -- **Default is BYPASS, not ENABLED**
- `session_id: str = None` -- persist browser context

**Page Behavior:**
- `wait_until: str = "domcontentloaded"`
- `page_timeout: int = 60000` (ms)
- `wait_for: str = None` -- CSS selector or JS condition
- `js_code: Union[str, List[str]] = None`
- `scan_full_page: bool = False`
- `scroll_delay: float = 0.2`
- `delay_before_return_html: float = 0.1`

**Anti-Detection:**
- `simulate_user: bool = False`
- `override_navigator: bool = False`
- `magic: bool = False`

**Deep Crawling:**
- `deep_crawl_strategy: Optional[DeepCrawlStrategy] = None`

**Streaming:**
- `stream: bool = False` -- enables streaming in arun_many and deep crawl

**Concurrency:**
- `semaphore_count: int = 5`
- `mean_delay: float = 0.1`
- `max_range: float = 0.3`

**Link Filtering:**
- `exclude_external_links: bool = False`
- `exclude_internal_links: bool = False`
- `exclude_social_media_links: bool = False`
- `exclude_domains: list = []`

**Media:**
- `screenshot: bool = False`
- `pdf: bool = False`

### clone() Method

Creates a new CrawlerRunConfig with selective overrides:
```python
new_config = config.clone(stream=True, cache_mode=CacheMode.BYPASS)
```

---

## 8. CacheMode

### Import Path
```python
from crawl4ai import CacheMode
```

### Values
| Mode | Value | Behavior |
|------|-------|----------|
| `ENABLED` | "enabled" | Normal caching (read and write) |
| `DISABLED` | "disabled" | No caching at all |
| `READ_ONLY` | "read_only" | Only read from cache, don't write |
| `WRITE_ONLY` | "write_only" | Only write to cache, don't read |
| `BYPASS` | "bypass" | Bypass cache for this operation |

**Important:** `CrawlerRunConfig` defaults to `CacheMode.BYPASS`. If `cache_mode` is `None` at arun() time, it falls back to `CacheMode.ENABLED`.

---

## 9. arun() and arun_many()

### arun()

```python
async def arun(
    self,
    url: str,
    config: CrawlerRunConfig = None,
    **kwargs,
) -> RunManyReturn:
```

- Takes a single URL and a CrawlerRunConfig
- Returns a `CrawlResult` normally
- When `config.deep_crawl_strategy` is set, returns `List[CrawlResult]` (batch) or `AsyncGenerator[CrawlResult]` (stream)
- Auto-starts the crawler if not already started

### arun_many()

```python
async def arun_many(
    self,
    urls: List[str],
    config: Optional[Union[CrawlerRunConfig, List[CrawlerRunConfig]]] = None,
    dispatcher: Optional[BaseDispatcher] = None,
    **kwargs,
) -> RunManyReturn:
```

- Takes multiple URLs
- `config` can be a single config (used for all URLs) or a list of configs with url_matcher for URL-specific settings
- Default dispatcher: `MemoryAdaptiveDispatcher` with RateLimiter(base_delay=(1.0, 3.0), max_delay=60.0, max_retries=3)

### Streaming Support in arun_many()

**YES, streaming is supported.** Set `config.stream = True`:

```python
# Batch mode (default)
results = await crawler.arun_many(urls=urls, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS))
for result in results:
    print(result.url)

# Streaming mode
config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=True)
async for result in await crawler.arun_many(urls=urls, config=config):
    print(f"Got result for {result.url}")
```

When streaming, `arun_many` returns an async generator, not a list.

### Dispatchers

```python
from crawl4ai import MemoryAdaptiveDispatcher, SemaphoreDispatcher, RateLimiter

# Memory-adaptive (default)
dispatcher = MemoryAdaptiveDispatcher(
    rate_limiter=RateLimiter(base_delay=(1.0, 3.0), max_delay=60.0, max_retries=3),
)

# Simple semaphore-based
dispatcher = SemaphoreDispatcher(max_concurrent=10)
```

---

## 10. Deep Crawling

### Import Paths
```python
from crawl4ai import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
)
```

### BFSDeepCrawlStrategy (Primary Strategy)

```python
class BFSDeepCrawlStrategy(DeepCrawlStrategy):
    def __init__(
        self,
        max_depth: int,                              # REQUIRED - max crawl depth
        filter_chain: FilterChain = FilterChain(),    # URL filters
        url_scorer: Optional[URLScorer] = None,       # URL scoring
        include_external: bool = False,               # follow external links
        score_threshold: float = -inf,                # minimum score to follow
        max_pages: int = inf,                         # max total pages to crawl
        logger: Optional[logging.Logger] = None,
        resume_state: Optional[Dict] = None,          # for crash recovery
        on_state_change: Optional[Callable] = None,   # state change callback
    )
```

### How Deep Crawling Works

1. **DeepCrawlDecorator** wraps `arun()` at crawler init time (line 167-168 in async_webcrawler.py)
2. When `config.deep_crawl_strategy` is set, the decorator intercepts `arun()` and delegates to the strategy's `arun()` method
3. The strategy internally uses `crawler.arun_many()` for each BFS level
4. For each level, it clones the config with `deep_crawl_strategy=None` to prevent recursion
5. After each level completes, it discovers links from results and builds the next level

### URL Deduplication

- The strategy maintains a `visited` set internally
- URLs are normalized using `normalize_url_for_deep_crawl()` before checking
- Fragment stripping is applied
- **There is no way to provide an external set of "already visited" URLs** at construction time (unless using `resume_state`)
- `resume_state` can contain a `visited` list for crash recovery

### Link Discovery

Links are extracted from `result.links["internal"]` (and `result.links["external"]` if `include_external=True`).

For each link:
1. URL is normalized
2. Checked against `visited` set
3. Passed through `filter_chain` (except depth 0)
4. Optionally scored via `url_scorer`
5. Checked against `score_threshold`
6. If `max_pages` limit would be exceeded, top-scored URLs are selected

### Streaming in Deep Crawl

Deep crawl supports streaming. Set `config.stream = True` and the results will be yielded as they are crawled:

```python
config = CrawlerRunConfig(
    deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=2, max_pages=50),
    stream=True,
    cache_mode=CacheMode.BYPASS,
)

async for result in await crawler.arun(url="https://example.com", config=config):
    print(f"Depth {result.metadata.get('depth')}: {result.url}")
```

### Deep Crawl + LLM Extraction

**YES, this works.** The extraction strategy is applied to each page during the crawl because each page goes through the normal `arun()` pipeline:

```python
config = CrawlerRunConfig(
    extraction_strategy=LLMExtractionStrategy(
        llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token="sk-..."),
        instruction="Extract all lawyer names and emails.",
        schema=schema,
    ),
    deep_crawl_strategy=BFSDeepCrawlStrategy(
        max_depth=2,
        max_pages=20,
        filter_chain=FilterChain([
            DomainFilter(allowed_domains=["example.com"]),
        ]),
    ),
    cache_mode=CacheMode.BYPASS,
    stream=True,
)
```

### Filters

**FilterChain** -- runs all filters; URL must pass all of them:
```python
chain = FilterChain([
    DomainFilter(allowed_domains=["example.com"]),
    URLPatternFilter(patterns=["/team/*", "/about/*"]),
    ContentTypeFilter(allowed_types=["text/html"]),
])
```

**DomainFilter:**
```python
DomainFilter(
    allowed_domains: Union[str, List[str]] = None,   # only allow these domains
    blocked_domains: Union[str, List[str]] = None,    # block these domains
)
```
Supports subdomain matching (e.g., `"example.com"` matches `"www.example.com"`).

**URLPatternFilter:**
```python
URLPatternFilter(
    patterns: Union[str, Pattern, List[Union[str, Pattern]]],
    use_glob: bool = True,
    reverse: bool = False,    # if True, EXCLUDE matching URLs
)
```
Supports: `*.html` (suffix), `/foo/*` (prefix), `*.example.com` (domain), glob patterns, regex (starts with `^` or ends with `$`).

**ContentTypeFilter:**
```python
ContentTypeFilter(
    allowed_types: Union[str, List[str]],    # e.g., ["text/html"]
    check_extension: bool = True,
)
```
Checks URL file extensions against MIME type mapping.

### No Adaptive Stop Condition

There is NO built-in mechanism to stop a deep crawl when "enough data has been found." The `max_pages` limit is the only stopping mechanism besides `max_depth`.

**For custom stop logic, you would need to:**
1. Use streaming mode
2. Accumulate results externally
3. Call `strategy.shutdown()` when satisfied (sets an internal cancel event)

```python
strategy = BFSDeepCrawlStrategy(max_depth=3, max_pages=100)
config = CrawlerRunConfig(deep_crawl_strategy=strategy, stream=True)

results = []
async for result in await crawler.arun(url=start_url, config=config):
    results.append(result)
    if all_fields_found(results):
        await strategy.shutdown()  # signals the cancel event
        break
```

### State Export / Resume

```python
strategy = BFSDeepCrawlStrategy(max_depth=3)
# ... after crawling ...
state = strategy.export_state()
# state = {"strategy_type": "bfs", "visited": [...], "pending": [...], "depths": {...}, "pages_crawled": N}

# Resume later:
resumed = BFSDeepCrawlStrategy(max_depth=3, resume_state=state)
```

---

## 11. Content Pipeline and Markdown Generation

### Pipeline Flow

```
Raw HTML
  --> ContentScrapingStrategy (LXMLWebScrapingStrategy)
      --> cleaned_html, links, media, metadata
  --> MarkdownGenerationStrategy (DefaultMarkdownGenerator)
      --> raw_markdown, markdown_with_citations, references_markdown
      --> fit_markdown (only if content_filter provided)
  --> ExtractionStrategy (LLMExtractionStrategy, etc.)
      --> extracted_content (JSON string)
```

### fit_markdown

`fit_markdown` is only generated when a content filter is provided to the `DefaultMarkdownGenerator`:

```python
from crawl4ai import DefaultMarkdownGenerator, PruningContentFilter

md_gen = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter()  # or BM25ContentFilter(user_query="...")
)

config = CrawlerRunConfig(
    markdown_generator=md_gen,
    extraction_strategy=LLMExtractionStrategy(
        llm_config=llm_config,
        input_format="fit_markdown",  # now this will work
    ),
)
```

Without a content filter, `fit_markdown` is empty string, and `input_format="fit_markdown"` falls back to `"markdown"`.

---

## 12. Resource Lifecycle

### Browser Mode (AsyncPlaywrightCrawlerStrategy)

```
AsyncWebCrawler.__init__() --> creates strategy, DeepCrawlDecorator
AsyncWebCrawler.start()    --> strategy.__aenter__() --> launches Playwright browser
  arun() / arun_many()     --> uses existing browser, creates new pages per request
AsyncWebCrawler.close()    --> strategy.__aexit__() --> closes browser
```

### HTTP Mode (AsyncHTTPCrawlerStrategy)

```
AsyncWebCrawler.__init__() --> creates strategy
AsyncWebCrawler.start()    --> strategy.start() --> creates aiohttp.ClientSession
  arun() / arun_many()     --> uses existing session for all requests
AsyncWebCrawler.close()    --> strategy.close() --> closes session
```

### Context Manager Pattern (Recommended)

```python
async with AsyncWebCrawler(config=browser_config) as crawler:
    # Browser/session is open here
    result = await crawler.arun(url="...")
# Browser/session is automatically closed
```

### Explicit Lifecycle (for Long-Running)

```python
crawler = AsyncWebCrawler(config=browser_config)
await crawler.start()
# ... many arun() calls ...
await crawler.close()
```

### Auto-Start

If you call `arun()` without `start()`, it auto-starts:
```python
crawler = AsyncWebCrawler()
result = await crawler.arun(url="...")  # auto-calls start()
await crawler.close()  # still need to close manually
```

---

## 13. Verified Import Paths

All imports below have been verified by actually running them in Python 3.14:

```python
# Core crawler
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# HTTP-only mode
from crawl4ai import HTTPCrawlerConfig
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy  # NOT in __init__

# LLM extraction
from crawl4ai import LLMExtractionStrategy, LLMConfig

# Other extraction strategies
from crawl4ai import (
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
    CosineStrategy,
)

# Result model
from crawl4ai import CrawlResult

# Deep crawling
from crawl4ai import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    DeepCrawlStrategy,
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
    ContentRelevanceFilter,
    SEOFilter,
)

# Scorers
from crawl4ai import (
    KeywordRelevanceScorer,
    CompositeScorer,
    DomainAuthorityScorer,
    FreshnessScorer,
    PathDepthScorer,
)

# Dispatchers
from crawl4ai import MemoryAdaptiveDispatcher, SemaphoreDispatcher, RateLimiter

# Content processing
from crawl4ai import DefaultMarkdownGenerator, PruningContentFilter, BM25ContentFilter

# Adaptive crawler
from crawl4ai import AdaptiveCrawler, AdaptiveConfig
```

---

## 14. Gotchas and Version-Specific Issues

### 1. AsyncHTTPCrawlerStrategy is NOT in the public API

It is not exported from `crawl4ai.__init__`. You must import from `crawl4ai.async_crawler_strategy`. This is likely intentional as it's considered lower-level.

### 2. HTTPCrawlerConfig parameter naming confusion

`AsyncHTTPCrawlerStrategy.__init__` takes `browser_config` parameter but expects `HTTPCrawlerConfig`, not `BrowserConfig`. This is a naming inconsistency in the source.

### 3. Deprecated parameters on LLMExtractionStrategy raise errors

Setting `provider`, `api_token`, `base_url`, or `api_base` directly will raise `AttributeError` via `__setattr__` override. Always use `llm_config=LLMConfig(...)`.

### 4. CrawlerRunConfig defaults to CacheMode.BYPASS

This means every request fetches fresh content by default. If you want caching, explicitly set `cache_mode=CacheMode.ENABLED`.

### 5. extracted_content is a JSON string

`result.extracted_content` is a `str` containing JSON, not a parsed dict/list. Always `json.loads()` it.

### 6. fit_markdown requires content_filter

`input_format="fit_markdown"` only works if you configure `DefaultMarkdownGenerator(content_filter=...)`. Otherwise it silently falls back to regular markdown.

### 7. Deep crawl always processes starting URL without filtering

The `can_process_url` method bypasses the filter chain for depth 0 (the starting URL). This is by design.

### 8. Deep crawl uses arun_many internally

Each BFS level uses `arun_many()` with the default dispatcher. The config is cloned with `deep_crawl_strategy=None` to prevent recursion.

### 9. The markdown property is a special StringCompatibleMarkdown

`result.markdown` behaves like a string (returns raw_markdown) but also has attributes like `.fit_markdown`, `.markdown_with_citations`, etc. This dual nature can be confusing.

### 10. proxy parameter on BrowserConfig is deprecated

Use `proxy_config` instead. Setting both will cause `proxy_config` to take precedence.

### 11. LLMExtractionStrategy uses litellm

The actual LLM calls go through `litellm` via `perform_completion_with_backoff()`. The provider string format follows litellm conventions (e.g., "openai/gpt-4o-mini").

### 12. config.py loads .env via python-dotenv

`config.py` calls `load_dotenv()` at import time. This means environment variables from `.env` files are automatically loaded when crawl4ai is imported.

### 13. Default word_count_threshold is 1

`MIN_WORD_THRESHOLD = 1` in config.py. This is very permissive -- almost nothing gets filtered out.

---

## 15. Recommendations for the Scraper Project

### Single-URL Mode

```python
# For simple pages (no JS rendering needed):
http_strategy = AsyncHTTPCrawlerStrategy(browser_config=HTTPCrawlerConfig())
crawler = AsyncWebCrawler(crawler_strategy=http_strategy)

# For JS-heavy pages:
crawler = AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False))
```

Use `LLMExtractionStrategy` with appropriate `input_format` based on content type:
- Static content-heavy pages: `input_format="markdown"` (default, best token efficiency)
- Structured data pages: `input_format="html"` or `input_format="cleaned_html"`

### Multi-URL Mode

Use `arun_many()` with `stream=True` for yielding results as they complete:

```python
config = CrawlerRunConfig(
    extraction_strategy=strategy,
    cache_mode=CacheMode.BYPASS,
    stream=True,
)

async for result in await crawler.arun_many(urls=url_list, config=config):
    yield result  # stream to caller
```

### Crawl Mode

Use `BFSDeepCrawlStrategy` with appropriate filters:

```python
from urllib.parse import urlparse

domain = urlparse(start_url).netloc

strategy = BFSDeepCrawlStrategy(
    max_depth=3,
    max_pages=100,
    filter_chain=FilterChain([
        DomainFilter(allowed_domains=[domain]),
        ContentTypeFilter(allowed_types=["text/html"]),
    ]),
)

config = CrawlerRunConfig(
    extraction_strategy=llm_strategy,
    deep_crawl_strategy=strategy,
    cache_mode=CacheMode.BYPASS,
    stream=True,
)

async for result in await crawler.arun(url=start_url, config=config):
    # Process each page as it's crawled
    if result.success and result.extracted_content:
        data = json.loads(result.extracted_content)
        # Accumulate results, check if done
```

For custom stop conditions, keep a reference to the strategy and call `strategy.shutdown()` when done.

### Architecture Recommendation

Create **two crawler instances** at startup:
1. `http_crawler` with `AsyncHTTPCrawlerStrategy` for fast HTTP-only fetching
2. `browser_crawler` with `AsyncPlaywrightCrawlerStrategy` for JS-rendered pages

Route URLs to the appropriate crawler based on the fetching mode requested by the user. Both can share the same `CrawlerRunConfig` (including extraction strategy).

### LLM Configuration Recommendation

Always explicitly pass `api_token` to `LLMConfig`. Do not rely on environment variable auto-detection as it adds complexity and potential failure modes:

```python
llm_config = LLMConfig(
    provider="openai/gpt-4o-mini",
    api_token=os.environ["OPENAI_API_KEY"],
)
```

---

## Source File Reference

| File | Contains |
|------|----------|
| `__init__.py` | Public API exports |
| `async_configs.py` | BrowserConfig, HTTPCrawlerConfig, CrawlerRunConfig, LLMConfig, ProxyConfig |
| `async_webcrawler.py` | AsyncWebCrawler, arun(), arun_many() |
| `async_crawler_strategy.py` | AsyncPlaywrightCrawlerStrategy, AsyncHTTPCrawlerStrategy |
| `extraction_strategy.py` | LLMExtractionStrategy, ExtractionStrategy base, CosineStrategy, JsonCss/XPath strategies |
| `models.py` | CrawlResult, MarkdownGenerationResult, Link, Links, TokenUsage |
| `cache_context.py` | CacheMode enum, CacheContext |
| `prompts.py` | LLM prompt templates |
| `config.py` | DEFAULT_PROVIDER, PROVIDER_MODELS_PREFIXES, constants |
| `deep_crawling/base_strategy.py` | DeepCrawlStrategy base, DeepCrawlDecorator |
| `deep_crawling/bfs_strategy.py` | BFSDeepCrawlStrategy |
| `deep_crawling/dfs_strategy.py` | DFSDeepCrawlStrategy |
| `deep_crawling/bff_strategy.py` | BestFirstCrawlingStrategy |
| `deep_crawling/filters.py` | FilterChain, URLPatternFilter, DomainFilter, ContentTypeFilter, SEOFilter |
| `deep_crawling/scorers.py` | URLScorer variants |
| `markdown_generation_strategy.py` | DefaultMarkdownGenerator |
| `content_filter_strategy.py` | PruningContentFilter, BM25ContentFilter, LLMContentFilter |
| `types.py` | Type aliases, create_llm_config() |
| `async_dispatcher.py` | MemoryAdaptiveDispatcher, SemaphoreDispatcher, RateLimiter |
