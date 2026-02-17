# scraper

[![Python versions](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/promisingcoder/generic-scraper-1/blob/main/LICENSE)

Extract structured data from any website using LLMs. Powered by [crawl4ai](https://github.com/unclecode/crawl4ai).

Define the fields you want, point it at a URL, and get back clean, typed data. Supports single-page extraction, multi-URL batch processing, and full-site crawling with automatic link discovery.

## Installation

```bash
# Basic install (HTTP backend only)
pip install .

# With browser support (Playwright, for JS-rendered pages)
pip install ".[browser]"

# Development install (editable)
pip install -e ".[dev]"
```

### Environment Setup

Set your OpenAI API key (or any litellm-compatible provider key):

```bash
export OPENAI_API_KEY="sk-..."

# Optional: backup keys for automatic rotation on rate limits
export OPENAI_API_KEY_BACKUP_1="sk-..."
export OPENAI_API_KEY_BACKUP_2="sk-..."
```

Or copy `.env.example` to `.env` and fill in your keys.

## Quick Start -- Python API

### Single URL

```python
import asyncio
from scraper import Scraper, define_fields

fields = define_fields(
    ("company_name", "The name of the company"),
    ("email", "Contact email address"),
    ("phone", "Main phone number"),
)

async def main():
    async with Scraper() as s:
        result = await s.scrape("https://example.com", fields)
        print(result.data)

asyncio.run(main())
```

Or use the synchronous helper for simple scripts:

```python
from scraper import scrape_sync, define_fields

fields = define_fields(
    ("title", "The page title"),
    ("description", "The page description"),
)
result = scrape_sync("https://example.com", fields)
print(result.data)
```

### Multi-URL Streaming

```python
import asyncio
from scraper import Scraper, define_fields

fields = define_fields(("title", "The page title"))

async def main():
    async with Scraper() as s:
        async for result in s.scrape_many(
            ["https://example.com", "https://example.org"],
            fields,
        ):
            print(result.url, result.data)

asyncio.run(main())
```

### Crawl Mode

```python
import asyncio
from scraper import Scraper, define_fields

fields = define_fields(
    ("company_name", "Name of the company"),
    ("ceo", "Name of the CEO"),
    ("emails", "All contact email addresses", "list[string]"),
)

async def main():
    async with Scraper() as s:
        async for result in s.crawl("https://example.com", fields, max_pages=20):
            print(f"[{result.url}] {result.data}")

asyncio.run(main())
```

## Quick Start -- CLI

The `scraper` command has three subcommands: `scrape`, `scrape-many`, and `crawl`.

```bash
# Single URL
scraper scrape https://example.com \
  --field "company_name:The name of the company" \
  --field "email:Contact email address"

# Multiple URLs
scraper scrape-many https://example.com https://example.org \
  --field "title:The page title"

# Crawl a site
scraper crawl https://example.com \
  --field "company_name:Name of the company" \
  --field "emails:All contact emails:list[string]" \
  --max-pages 20
```

Common options (placed before the subcommand):

```bash
scraper --backend browser scrape https://example.com --field "title:Page title"
scraper --model openai/gpt-4o scrape https://example.com --field "title:Page title"
```

Run `scraper --help` or `scraper <subcommand> --help` for full usage.

## Field Definitions

Fields are defined as tuples of `(name, description)` or `(name, description, type)`.

| Type | Description | Example Value | Crawl Behavior |
|------|-------------|---------------|----------------|
| `string` | Single text value | `"Acme Corp"` | Bounded -- keeps first non-null value found |
| `number` | Numeric value | `42.5` | Bounded -- keeps first non-null value found |
| `integer` | Integer value | `100` | Bounded -- keeps first non-null value found |
| `boolean` | True/false | `true` | Bounded -- keeps first non-null value found |
| `list[string]` | List of strings | `["a@b.com", "c@d.com"]` | Unbounded -- unions across pages |
| `list[number]` | List of numbers | `[1, 2, 3]` | Unbounded -- unions across pages |
| `list[object]` | List of objects | `[{"name": "Alice"}]` | Unbounded -- unions across pages |
| `list[boolean]` | List of booleans | `[true, false]` | Unbounded -- unions across pages |

**Bounded fields** (scalars) lock in the first non-null value found during a crawl. Once all bounded fields are filled and there are no unbounded fields, crawling stops early.

**Unbounded fields** (lists) are unioned and deduplicated across all crawled pages. Because the total count is unknown, crawling continues until `max_pages` is reached or no more links remain.

## Fetching Backends

| Backend | Flag | Best For | Speed | JS Support |
|---------|------|----------|-------|------------|
| `http` | `--backend http` (default) | Static/server-rendered pages | Fast | No |
| `browser` | `--backend browser` | JS-heavy SPAs, dynamically loaded content | Slower | Yes |

The HTTP backend uses `crawl4ai`'s async HTTP strategy (aiohttp). The browser backend launches a headless Chromium via Playwright.

Install browser support with `pip install ".[browser]"` and run `playwright install chromium` before first use.

## Crawl Mode Details

Crawl mode starts from a URL, discovers internal links (same domain), and extracts from each page. Key behaviors:

- **Link prioritization**: High-value paths (`/about`, `/contact`, `/team`, etc.) are crawled first.
- **Deduplication**: URLs are normalized and deduplicated so each page is visited only once.
- **Early stopping**: If all fields are bounded (scalar) and all have been filled, crawling stops immediately.
- **Max limits**: Crawling respects `max_pages` (default 50) and `max_depth` (default 3).

Stopping reasons:
- `all_fields_found` -- all scalar fields filled, no list fields present
- `max_pages_reached` -- hit the page limit
- `no_more_links` -- no more unvisited internal links

## Configuration

### Python API

```python
Scraper(
    backend="http",          # "http" or "browser"
    model="openai/gpt-4o-mini",  # any litellm-compatible model
    max_concurrent=5,        # concurrent requests (scrape_many)
    timeout=30000,           # page load timeout in ms
    headless=True,           # browser only: headless mode
    stealth=True,            # browser only: anti-bot stealth
    max_pages=50,            # crawl mode: page limit
    max_depth=3,             # crawl mode: link depth limit
    input_format="markdown", # content format for LLM
    verbose=False,           # crawl4ai debug logging
)
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Primary API key for the LLM provider |
| `OPENAI_API_KEY_BACKUP_1` | No | First backup key (auto-rotated on 429/auth errors) |
| `OPENAI_API_KEY_BACKUP_2` | No | Second backup key |

## License

MIT
