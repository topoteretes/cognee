"""BeautifulSoup-based web crawler for extracting content from web pages.

This module provides the BeautifulSoupCrawler class for fetching and extracting content
from web pages using BeautifulSoup or Playwright for JavaScript-rendered pages. It
supports robots.txt handling, rate limiting, and custom extraction rules.
"""

from typing import Union, Dict, Any, Optional, List
from dataclasses import dataclass
from bs4 import BeautifulSoup
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionRule:
    """Normalized extraction rule for web content.

    Attributes:
        selector: CSS selector for extraction (if any).
        xpath: XPath expression for extraction (if any).
        attr: HTML attribute to extract (if any).
        all: If True, extract all matching elements; otherwise, extract first.
        join_with: String to join multiple extracted elements.
    """

    selector: Optional[str] = None
    xpath: Optional[str] = None
    attr: Optional[str] = None
    all: bool = False
    join_with: str = " "


class BeautifulSoupLoader(LoaderInterface):
    """Crawler for fetching and extracting web content using BeautifulSoup.

    Supports asynchronous HTTP requests, Playwright for JavaScript rendering, robots.txt
    compliance, and rate limiting. Extracts content using CSS selectors or XPath rules.

    Attributes:
        concurrency: Number of concurrent requests allowed.
        crawl_delay: Minimum seconds between requests to the same domain.
        max_crawl_delay: Maximum crawl delay to respect from robots.txt (None = no limit).
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries for failed requests.
        retry_delay_factor: Multiplier for exponential backoff on retries.
        headers: HTTP headers for requests (e.g., User-Agent).
        robots_cache_ttl: Time-to-live for robots.txt cache in seconds.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["html"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/html", "text/plain"]

    @property
    def loader_name(self) -> str:
        return "beautiful_soup_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        can = extension in self.supported_extensions and mime_type in self.supported_mime_types
        return can

    def _get_default_extraction_rules(self):
        # Comprehensive default extraction rules for common HTML content
        return {
            # Meta information
            "title": {"selector": "title", "all": False},
            "meta_description": {
                "selector": "meta[name='description']",
                "attr": "content",
                "all": False,
            },
            "meta_keywords": {
                "selector": "meta[name='keywords']",
                "attr": "content",
                "all": False,
            },
            # Open Graph meta tags
            "og_title": {
                "selector": "meta[property='og:title']",
                "attr": "content",
                "all": False,
            },
            "og_description": {
                "selector": "meta[property='og:description']",
                "attr": "content",
                "all": False,
            },
            # Main content areas (prioritized selectors)
            "article": {"selector": "article", "all": True, "join_with": "\n\n"},
            "main": {"selector": "main", "all": True, "join_with": "\n\n"},
            # Semantic content sections
            "headers_h1": {"selector": "h1", "all": True, "join_with": "\n"},
            "headers_h2": {"selector": "h2", "all": True, "join_with": "\n"},
            "headers_h3": {"selector": "h3", "all": True, "join_with": "\n"},
            "headers_h4": {"selector": "h4", "all": True, "join_with": "\n"},
            "headers_h5": {"selector": "h5", "all": True, "join_with": "\n"},
            "headers_h6": {"selector": "h6", "all": True, "join_with": "\n"},
            # Text content
            "paragraphs": {"selector": "p", "all": True, "join_with": "\n\n"},
            "blockquotes": {"selector": "blockquote", "all": True, "join_with": "\n\n"},
            "preformatted": {"selector": "pre", "all": True, "join_with": "\n\n"},
            # Lists
            "ordered_lists": {"selector": "ol", "all": True, "join_with": "\n"},
            "unordered_lists": {"selector": "ul", "all": True, "join_with": "\n"},
            "list_items": {"selector": "li", "all": True, "join_with": "\n"},
            "definition_lists": {"selector": "dl", "all": True, "join_with": "\n"},
            # Tables
            "tables": {"selector": "table", "all": True, "join_with": "\n\n"},
            "table_captions": {
                "selector": "caption",
                "all": True,
                "join_with": "\n",
            },
            # Code blocks
            "code_blocks": {"selector": "code", "all": True, "join_with": "\n"},
            # Figures and media descriptions
            "figures": {"selector": "figure", "all": True, "join_with": "\n\n"},
            "figcaptions": {"selector": "figcaption", "all": True, "join_with": "\n"},
            "image_alts": {"selector": "img", "attr": "alt", "all": True, "join_with": " "},
            # Links (text content, not URLs to avoid clutter)
            "link_text": {"selector": "a", "all": True, "join_with": " "},
            # Emphasized text
            "strong": {"selector": "strong", "all": True, "join_with": " "},
            "emphasis": {"selector": "em", "all": True, "join_with": " "},
            "marked": {"selector": "mark", "all": True, "join_with": " "},
            # Time and data elements
            "time": {"selector": "time", "all": True, "join_with": " "},
            "data": {"selector": "data", "all": True, "join_with": " "},
            # Sections and semantic structure
            "sections": {"selector": "section", "all": True, "join_with": "\n\n"},
            "asides": {"selector": "aside", "all": True, "join_with": "\n\n"},
            "details": {"selector": "details", "all": True, "join_with": "\n"},
            "summary": {"selector": "summary", "all": True, "join_with": "\n"},
            # Navigation (may contain important links/structure)
            "nav": {"selector": "nav", "all": True, "join_with": "\n"},
            # Footer information
            "footer": {"selector": "footer", "all": True, "join_with": "\n"},
            # Divs with specific content roles
            "content_divs": {
                "selector": "div[role='main'], div[role='article'], div.content, div#content",
                "all": True,
                "join_with": "\n\n",
            },
        }

    async def load(
        self,
        file_path: str,
        extraction_rules: dict[str, Any] = None,
        join_all_matches: bool = False,
        **kwargs,
    ):
        """Load an HTML file, extract content, and save to storage.

        Args:
            file_path: Path to the HTML file
            extraction_rules: Dict of CSS selector rules for content extraction
            join_all_matches: If True, extract all matching elements for each rule
            **kwargs: Additional arguments

        Returns:
            Path to the stored extracted text file
        """
        if extraction_rules is None:
            extraction_rules = self._get_default_extraction_rules()
            logger.info("Using default comprehensive extraction rules for HTML content")

        logger.info(f"Processing HTML file: {file_path}")

        from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
        from cognee.infrastructure.files.storage import get_file_storage, get_storage_config

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
            f.seek(0)
            html = f.read()

        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

        # Normalize extraction rules
        normalized_rules: List[ExtractionRule] = []
        for _, rule in extraction_rules.items():
            r = self._normalize_rule(rule)
            if join_all_matches:
                r.all = True
            normalized_rules.append(r)

        pieces = []
        for rule in normalized_rules:
            text = self._extract_from_html(html, rule)
            if text:
                pieces.append(text)

        full_content = " ".join(pieces).strip()

        # remove after defaults for extraction rules
        # Fallback: If no content extracted, check if the file is plain text (not HTML)
        if not full_content:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            # If there are no HTML tags, treat as plain text
            if not soup.find():
                logger.warning(
                    f"No HTML tags found in {file_path}. Treating as plain text. "
                    "This may happen when content is pre-extracted (e.g., via Tavily with text format)."
                )
                full_content = html.decode("utf-8") if isinstance(html, bytes) else html
                full_content = full_content.strip()

        if not full_content:
            logger.warning(f"No content extracted from HTML file: {file_path}")

        # Store the extracted content
        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, full_content)

        logger.info(f"Extracted {len(full_content)} characters from HTML")
        return full_file_path

    def _normalize_rule(self, rule: Union[str, Dict[str, Any]]) -> ExtractionRule:
        """Normalize an extraction rule to an ExtractionRule dataclass.

        Args:
            rule: A string (CSS selector) or dict with extraction parameters.

        Returns:
            ExtractionRule: Normalized extraction rule.

        Raises:
            ValueError: If the rule is invalid.
        """
        if isinstance(rule, str):
            return ExtractionRule(selector=rule)
        if isinstance(rule, dict):
            return ExtractionRule(
                selector=rule.get("selector"),
                xpath=rule.get("xpath"),
                attr=rule.get("attr"),
                all=bool(rule.get("all", False)),
                join_with=rule.get("join_with", " "),
            )
        raise ValueError(f"Invalid extraction rule: {rule}")

    def _extract_from_html(self, html: str, rule: ExtractionRule) -> str:
        """Extract content from HTML using BeautifulSoup or lxml XPath.

        Args:
            html: The HTML content to extract from.
            rule: The extraction rule to apply.

        Returns:
            str: The extracted content.

        Raises:
            RuntimeError: If XPath is used but lxml is not installed.
        """
        soup = BeautifulSoup(html, "html.parser")

        if rule.xpath:
            try:
                from lxml import html as lxml_html
            except ImportError:
                raise RuntimeError(
                    "XPath requested but lxml is not available. Install lxml or use CSS selectors."
                )
            doc = lxml_html.fromstring(html)
            nodes = doc.xpath(rule.xpath)
            texts = []
            for n in nodes:
                if hasattr(n, "text_content"):
                    texts.append(n.text_content().strip())
                else:
                    texts.append(str(n).strip())
            return rule.join_with.join(t for t in texts if t)

        if not rule.selector:
            return ""

        if rule.all:
            nodes = soup.select(rule.selector)
            pieces = []
            for el in nodes:
                if rule.attr:
                    val = el.get(rule.attr)
                    if val:
                        pieces.append(val.strip())
                else:
                    text = el.get_text(strip=True)
                    if text:
                        pieces.append(text)
            return rule.join_with.join(pieces).strip()
        else:
            el = soup.select_one(rule.selector)
            if el is None:
                return ""
            if rule.attr:
                val = el.get(rule.attr)
                return (val or "").strip()
            return el.get_text(strip=True)
