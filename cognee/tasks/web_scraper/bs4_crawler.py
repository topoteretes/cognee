import asyncio
import time
from typing import Union, List, Dict, Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from playwright.async_api import async_playwright
except ImportError:
    logger.error(
        "Failed to import playwright, make sure to install using pip install playwright>=1.9.0"
    )
    async_playwright = None

try:
    from protego import Protego
except ImportError:
    logger.error("Failed to import protego, make sure to install using pip install protego>=0.1")
    Protego = None


class BeautifulSoupCrawler:
    def __init__(
        self,
        *,
        concurrency: int = 5,
        crawl_delay: float = 0.5,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_delay_factor: float = 0.5,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        concurrency: number of concurrent requests allowed
        crawl_delay: minimum seconds to wait between requests to the SAME domain
        timeout: per-request timeout
        max_retries: number of retries on network errors
        retry_delay_factor: multiplier for exponential retry failure delay
        headers: default headers for requests
        """
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.crawl_delay = crawl_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_factor = retry_delay_factor
        self.headers = headers or {"User-Agent": "Cognee-Scraper/1.0"}
        self._last_request_time_per_domain: Dict[str, float] = {}
        self._client: Optional[httpx.AsyncClient] = None

    # ---------- lifecycle helpers ----------
    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---------- rate limiting ----------
    def _domain_from_url(self, url: str) -> str:
        try:
            return urlparse(url).netloc
        except Exception:
            return url

    async def _respect_rate_limit(self, url: str, crawl_delay: Optional[float] = None):
        domain = self._domain_from_url(url)
        last = self._last_request_time_per_domain.get(domain)
        delay = crawl_delay if crawl_delay is not None else self.crawl_delay
        if last is None:
            self._last_request_time_per_domain[domain] = time.time()
            return
        elapsed = time.time() - last
        wait_for = delay - elapsed
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        self._last_request_time_per_domain[domain] = time.time()

    # ----------- robots.txt handling -----------
    async def _is_url_allowed(self, url: str) -> bool:
        if Protego is None:
            return True  # fallback if Protego not installed
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            content = await self._fetch_httpx(robots_url)
            rp = Protego.parse(content)
            agent = next((v for k, v in self.headers.items() if k.lower() == "user-agent"), "*")
            return rp.can_fetch(agent, url) or rp.can_fetch("*", url)
        except Exception:
            return True  # if no robots.txt, allow by default

    async def _get_crawl_delay(self, base_url: str) -> float:
        if Protego is None:
            return self.crawl_delay
        try:
            content = await self._fetch_httpx(base_url + "/robots.txt")
            rp = Protego.parse(content)
            agent = next((v for k, v in self.headers.items() if k.lower() == "user-agent"), "*")
            delay = rp.crawl_delay(agent) or rp.crawl_delay("*")
            return delay or self.crawl_delay
        except Exception:
            return self.crawl_delay

    # ---------- low-level fetchers ----------
    async def _fetch_httpx(self, url: str) -> str:
        await self._ensure_client()
        assert self._client is not None, "HTTP client not initialized"
        attempt = 0
        while True:
            try:
                # get crawl delay from robots.txt if available
                crawl_delay = await self._get_crawl_delay(
                    f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                )
                await self._respect_rate_limit(url, crawl_delay)
                resp = await self._client.get(url)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"Fetch failed for {url}: {exc}")
                    raise
                delay = self.retry_delay_factor * (2 ** (attempt - 1))
                logger.warning(f"Retrying {url} after {delay:.2f}s (attempt {attempt})")
                await asyncio.sleep(delay)

    async def _render_with_playwright(
        self, url: str, js_wait: float = 1.0, timeout: Optional[float] = None
    ) -> str:
        if async_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Install with `pip install playwright` and run `playwright install`."
            )
        attempt = 0
        while True:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    try:
                        context = await browser.new_context()
                        page = await context.new_page()
                        await page.goto(
                            url,
                            wait_until="networkidle",
                            timeout=int((timeout or self.timeout) * 1000),
                        )
                        if js_wait:
                            await asyncio.sleep(js_wait)
                        return await page.content()
                    finally:
                        await browser.close()
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"Playwright fetch failed for {url}: {exc}")
                    raise
                backoff = self.retry_delay_factor * (2 ** (attempt - 1))
                logger.warning(
                    f"Retrying playwright fetch {url} after {backoff:.2f}s (attempt {attempt})"
                )
                await asyncio.sleep(backoff)

    # ---------- extraction helpers ----------
    def _normalize_rule(self, rule) -> Dict[str, Any]:
        if isinstance(rule, str):
            return {"selector": rule, "attr": None, "all": False, "join_with": " "}
        if isinstance(rule, dict):
            return {
                "selector": rule.get("selector"),
                "attr": rule.get("attr"),
                "all": bool(rule.get("all")),
                "join_with": rule.get("join_with", " "),
                "xpath": rule.get("xpath"),
            }
        raise ValueError("Invalid extraction rule")

    def _extract_with_bs4(self, html: str, rule: Dict[str, Any]) -> str:
        soup = BeautifulSoup(html, "html.parser")
        sel = rule.get("selector")
        xpath = rule.get("xpath")
        attr = rule.get("attr")
        all_flag = rule.get("all", False)
        join_with = rule.get("join_with", " ")

        if xpath:
            try:
                from lxml import html as lxml_html
            except Exception:
                raise RuntimeError(
                    "XPath requested but lxml is not available. Install lxml or use CSS selectors."
                )
            doc = lxml_html.fromstring(html)
            nodes = doc.xpath(xpath)
            texts = []
            for n in nodes:
                if hasattr(n, "text_content"):
                    texts.append(n.text_content().strip())
                else:
                    texts.append(str(n).strip())
            return join_with.join(t for t in texts if t)
        else:
            if not sel:
                return ""
            if all_flag:
                nodes = soup.select(sel)
                pieces = []
                for el in nodes:
                    if attr:
                        val = el.get(attr)
                        if val:
                            pieces.append(val.strip())
                    else:
                        text = el.get_text(strip=True)
                        if text:
                            pieces.append(text)
                return join_with.join(pieces).strip()
            else:
                el = soup.select_one(sel)
                if el is None:
                    return ""
                if attr:
                    val = el.get(attr)
                    return (val or "").strip()
                return el.get_text(strip=True)

    # ---------- public methods ----------
    async def fetch_with_bs4(
        self,
        urls: Union[str, List[str]],
        extraction_rules: Dict[str, Any],
        *,
        use_playwright: bool = False,
        playwright_js_wait: float = 0.8,
        join_all_matches: bool = False,
        structured: bool = False,  # return structured output instead of concatenated string
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Fetch one or more URLs and extract text using BeautifulSoup (or lxml xpath).
        Returns: dict[url] -> concatenated string OR structured dict depending on `structured`.
        """
        if isinstance(urls, str):
            urls = [urls]

        # normalize rules
        normalized_rules = {}
        for field, rule in extraction_rules.items():
            r = self._normalize_rule(rule)
            if join_all_matches:
                r["all"] = True
            normalized_rules[field] = r

        async def _task(url: str):
            async with self._sem:
                allowed = await self._is_url_allowed(url)
                if not allowed:
                    logger.warning(f"URL disallowed by robots.txt: {url}")
                    return url, "" if not structured else {}

                # fetch (rendered or not)
                if use_playwright:
                    html = await self._render_with_playwright(
                        url, js_wait=playwright_js_wait, timeout=self.timeout
                    )
                else:
                    html = await self._fetch_httpx(url)

                if structured:
                    return url, {
                        field: self._extract_with_bs4(html, rule)
                        for field, rule in normalized_rules.items()
                    }

                pieces = []
                for field, rule in normalized_rules.items():
                    text = self._extract_with_bs4(html, rule)
                    if text:
                        pieces.append(text)
                concatenated = " ".join(pieces).strip()
                return url, concatenated

        tasks = [asyncio.create_task(_task(u)) for u in urls]
        results = {}
        for coro in asyncio.as_completed(tasks):
            try:
                url, text = await coro
            except Exception as e:
                results[url] = {} if structured else ""
                logger.error(f"Error processing {url}: {e}")
                continue
            results[url] = text
        return results
