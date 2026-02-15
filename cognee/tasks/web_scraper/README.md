# Web Scraper

## Overview

The `web_scraper` module provides tools for scraping web content and storing structured data in cognee's graph database. It supports two scraping backends:

1. **Default Crawler** - Uses `httpx` for HTTP requests with optional Playwright for JavaScript rendering
2. **Tavily API** - Uses Tavily for content extraction (requires `TAVILY_API_KEY` environment variable)

The module automatically selects the backend: Tavily if `TAVILY_API_KEY` is set, otherwise the default crawler.

## Components

### Functions

| Function | Description |
|----------|-------------|
| `fetch_page_content(urls)` | Fetches content from URLs, returns `dict[str, str]` mapping URLs to HTML |
| `web_scraper_task(url, ...)` | Scrapes URLs and stores `WebPage`, `WebSite`, `ScrapingJob` in graph database |
| `cron_web_scraper_task(url, schedule, ...)` | Schedules recurring scrapes with APScheduler cron expressions |

### Classes

| Class | Description |
|-------|-------------|
| `DefaultUrlCrawler` | Async crawler with concurrency, rate limiting, robots.txt compliance, Playwright support |
| `DefaultCrawlerConfig` | Configuration for default crawler (concurrency, timeouts, Playwright) |
| `TavilyConfig` | Configuration for Tavily API (API key, extract depth, timeout) |

### Data Models (extend `DataPoint`)

| Model | Key Fields |
|-------|------------|
| `WebPage` | `content`, `content_hash`, `scraped_at`, `status_code`, `page_size` |
| `WebSite` | `base_url`, `robots_txt`, `crawl_delay`, `page_count` |
| `ScrapingJob` | `urls`, `schedule`, `status`, `last_run`, `next_run` |

**Graph Relationships:** `ScrapingJob` → `WebSite` (`is_scraping`), `WebPage` → `WebSite` (`is_part_of`)

## Usage

### Basic URL Fetching

```python
from cognee.tasks.web_scraper import fetch_page_content

result = await fetch_page_content("https://example.com")
html = result["https://example.com"]
```

### Using DefaultUrlCrawler

```python
from cognee.tasks.web_scraper import DefaultUrlCrawler

async with DefaultUrlCrawler(concurrency=5, timeout=15.0) as crawler:
    results = await crawler.fetch_urls("https://example.com")
```

### Web Scraping Task with Graph Storage

```python
from cognee.tasks.web_scraper import web_scraper_task

graph_data = await web_scraper_task(
    url=["https://example.com"],
    job_name="my_scraping_job"
)
```

> [!NOTE]
> Requires APScheduler: `pip install APScheduler>=3.10`

### Scheduled Scraping

```python
from cognee.tasks.web_scraper import cron_web_scraper_task

await cron_web_scraper_task(
    url="https://example.com",
    schedule="0 0 * * *",
    job_name="daily_scrape"
)
```

### Integration with cognee.add()

```python
import cognee
from cognee.tasks.web_scraper.config import DefaultCrawlerConfig

config = DefaultCrawlerConfig(concurrency=5, timeout=15.0, use_playwright=False)
await cognee.add(data="https://example.com", soup_crawler_config=config)
await cognee.cognify()
```

## Configuration

### DefaultCrawlerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `concurrency` | `5` | Max concurrent requests |
| `crawl_delay` | `0.5` | Seconds between requests to same domain |
| `max_crawl_delay` | `10.0` | Max robots.txt crawl delay (None = no limit) |
| `timeout` | `15.0` | Request timeout in seconds |
| `max_retries` | `2` | Retry count for failed requests |
| `retry_delay_factor` | `0.5` | Exponential backoff multiplier |
| `headers` | `None` | Custom HTTP headers (defaults to Cognee User-Agent) |
| `use_playwright` | `False` | Enable JavaScript rendering |
| `playwright_js_wait` | `0.8` | JS wait time in seconds |
| `robots_cache_ttl` | `3600.0` | robots.txt cache TTL in seconds |
| `join_all_matches` | `False` | Join all CSS selector matches |

### TavilyConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `TAVILY_API_KEY` env | Tavily API key |
| `extract_depth` | `"basic"` | `"basic"` or `"advanced"` |
| `proxies` | `None` | Proxy configuration dict |
| `timeout` | `10` | Request timeout (1-60s) |

## Dependencies

**Required:** `httpx`, `pydantic`

**Optional:**
- `protego>=0.1` - robots.txt parsing
- `playwright>=1.9.0` - JS rendering
- `tavily-python>=0.7.0` - Tavily API
- `APScheduler>=3.10` - Cron scheduling

## Related

- [cognee docs](https://docs.cognee.ai) | [`cognee.tasks.ingestion`](../ingestion/) | [`DataPoint`](../../infrastructure/engine/models/DataPoint.py)
