import asyncio
import time
from typing import Union, List, Dict, Any, Optional
from bs4 import BeautifulSoup
import httpx
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from playwright.async_api import async_playwright
except ImportError:
    logger.error("Failed to import playwright, make sure to install using pip install playwright>=1.9.0")

try:
    from bs4 import BeautifulSoup
except ImportError:
    logger.error("Failed to import BeautifulSoup, make sure to install using pip install beautifulsoup4")
    




class BeautifulSoupCrawler:
    def __init__(
        self,
        *,
        concurrency: int = 5,
        delay_between_requests: float = 0.5,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_delay_factor: float = 0.5,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        concurrency: number of concurrent requests allowed
        delay_between_requests: minimum seconds to wait between requests to the SAME domain
        timeout: per-request timeout
        max_retries: number of retries on network errors
        retry_delay_factor: multiplier for exponential retry failure delay
        headers: default headers for requests
        use_httpx: require httpx for async HTTP. If not available, an informative error will be raised.
        """
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.delay_between_requests = delay_between_requests
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_factor = retry_delay_factor
        self.headers = headers or {"User-agent": "Cognee-Scraper/1.0"}
        self._last_request_time_per_domain: Dict[str, float] = {}
        self._client = None

    # ---------- lifecycle helpers ----------
    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers) if httpx else None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---------- rate limiting ----------
    def _domain_from_url(self, url: str) -> str:
        # quick parse to domain
        try:
            from urllib.parse import urlparse
            p = urlparse(url)
            return p.netloc
        except Exception:
            return url

    async def _respect_rate_limit(self, url: str):
        domain = self._domain_from_url(url)
        last = self._last_request_time_per_domain.get(domain)
        if last is None:
            self._last_request_time_per_domain[domain] = time.time()
            return
        elapsed = time.time() - last
        wait_for = self.delay_between_requests - elapsed
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        self._last_request_time_per_domain[domain] = time.time()
    
    # ----------- robots.txt handling could be added here -----------
    async def _is_url_allowed(self, url: str) -> bool:
        robots_txt_url = f"{self._get_base_url(url)}/robots.txt"
        robots_txt_content = await self._fetch_httpx(robots_txt_url)
        robots_txt_content = robots_txt_content.lower()
        user_agent_name = self.headers.get("User-agent")
        pos = robots_txt_content.find(f"user-agent: {user_agent_name}")
        if pos == -1:
            pos = robots_txt_content.find(f"user-agent:*")
        if pos == -1:
            return True
        
        pos = robots_txt_content.find("disallow", pos)
    # TODO: Research more about robots.txt format
    
        
          

    # ---------- low-level fetchers ----------
    async def _fetch_httpx(self, url: str) -> str:
        await self._ensure_client()
        assert self._client is not None, "HTTP client not initialized"
        attempt = 0
        while True:
            try:
                await self._respect_rate_limit(url)
                resp = await self._client.get(url)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = self.retry_delay_factor * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

    async def _render_with_playwright(self, url: str, js_wait: float = 1.0, timeout: Optional[float] = None) -> str:
        if async_playwright is None:
            raise RuntimeError("Playwright is not installed. Install with `pip install playwright` and run `playwright install`.")
        # Basic Playwright rendering (Chromium). This is slower but renders JS.
        attempt = 0
        while True:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context()
                    page = await context.new_page()
                    await page.goto(url, wait_until="networkidle", timeout=int((timeout or self.timeout) * 1000))
                    # optional short wait to let in-page JS mutate DOM
                    if js_wait:
                        await asyncio.sleep(js_wait)
                    content = await page.content()
                    await browser.close()
                    return content
            except Exception:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                backoff = self.backoff_factor * (2 ** (attempt - 1))
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
            # try lxml extraction via xpath if lxml is available
            try:
                from lxml import html as lxml_html
            except Exception:
                raise RuntimeError("XPath requested but lxml is not available. Install lxml or use CSS selectors.")
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

    # ---------- public API (keeps the signature you asked for) ----------
    async def fetch_with_bs4(
        self,
        urls: Union[str, List[str]],
        extraction_rules: Dict[str, Any],
        *,
        use_playwright: bool = False,
        playwright_js_wait: float = 0.8,
        join_all_matches: bool = False,  # if True, for each rule use all matches (join them)
    ) -> Dict[str, str]:
        """
        Fetch one or more URLs and extract text using BeautifulSoup (or lxml xpath).
        Returns: dict[url] -> single concatenated string (trimmed)
        Parameters:
          - urls: str or list[str]
          - extraction_rules: dict[field_name -> selector or rule-dict]
              rule-dict keys: selector (CSS), xpath (optional), attr (optional), all(bool), join_with(str)
          - use_playwright: if True, use Playwright to render JS (must be installed), otherwise normal fetch
          - playwright_js_wait: seconds to wait after load for JS to mutate DOM
          - join_all_matches: convenience: if True, treat each rule as all=True
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

        # concurrency control + gather tasks
        async def _task(url: str):
            async with self._sem:
                # fetch (rendered or not)
                if use_playwright:
                    html = await self._render_with_playwright(url, js_wait=playwright_js_wait, timeout=self.timeout)
                else:
                    html = await self._fetch_httpx(url)

                # Extract and concatenate results into a single string
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
                # store empty string on failure (or raise depending on your policy)
                results[url] = ""
                # Optionally you could log the error; for now we'll attach empty string
                continue
            results[url] = text
        return results
