import os
import hashlib
from datetime import datetime
from typing import Union, List
from urllib.parse import urlparse
from uuid import uuid5, NAMESPACE_OID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges
from cognee.modules.engine.operations.setup import setup

from .models import WebPage, WebSite, ScrapingJob
from .config import SoupCrawlerConfig, TavilyConfig
from .utils import fetch_page_content

logger = get_logger(__name__)


async def web_scraper_task(
    url: Union[str, List[str]],
    *,
    schedule: str = None,
    extraction_rules: dict = None,
    tavily_api_key: str = os.getenv("TAVILY_API_KEY"),
    soup_crawler_config: SoupCrawlerConfig = None,
    tavily_config: TavilyConfig = None,
    job_name: str = None,
):
    """
    Scrapes one or more URLs and returns WebPage, WebSite, and ScrapingJob data points.
    Unique IDs are assigned to each WebPage, WebSite, and ScrapingJob.
    Includes a description field summarizing other fields for each data point.
    """
    await setup()
    graph_db = await get_graph_engine()

    if isinstance(url, str):
        url = [url]

    soup_crawler_config, tavily_config, preferred_tool = check_arguments(
        tavily_api_key, extraction_rules, tavily_config, soup_crawler_config
    )
    now = datetime.now()
    job_name = job_name or f"scrape_{now.strftime('%Y%m%d_%H%M%S')}"
    status = "active"
    trigger = CronTrigger.from_crontab(schedule) if schedule else None
    next_run = trigger.get_next_fire_time(None, now) if trigger else None
    scraping_job_created = await graph_db.get_node(uuid5(NAMESPACE_OID, name=job_name))

    # Create description for ScrapingJob
    scraping_job_description = (
        f"Scraping job: {job_name}\n"
        f"URLs: {', '.join(url)}\n"
        f"Status: {status}\n"
        f"Schedule: {schedule}\n"
        f"Last run: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else 'Not scheduled'}"
    )

    scraping_job = ScrapingJob(
        id=uuid5(NAMESPACE_OID, name=job_name),
        name=job_name,
        urls=url,
        status=status,
        schedule=schedule,
        last_run=now,
        next_run=next_run,
        description=scraping_job_description,
    )

    if scraping_job_created:
        await graph_db.add_node(scraping_job)  # Update existing scraping job
    websites_dict = {}
    webpages = []

    # Fetch content
    results = await fetch_page_content(
        urls=url,
        preferred_tool=preferred_tool,
        tavily_config=tavily_config,
        soup_crawler_config=soup_crawler_config,
    )

    for page_url, content in results.items():
        parsed_url = urlparse(page_url)
        domain = parsed_url.netloc
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Create or update WebSite
        if base_url not in websites_dict:
            # Create description for WebSite
            website_description = (
                f"Website: {domain}\n"
                f"Base URL: {base_url}\n"
                f"Last crawled: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Page count: 1\n"
                f"Scraping tool: {preferred_tool}\n"
                f"Robots.txt: {'Available' if websites_dict.get(base_url, {}).get('robots_txt') else 'Not set'}\n"
                f"Crawl delay: 0.5 seconds"
            )

            websites_dict[base_url] = WebSite(
                id=uuid5(NAMESPACE_OID, name=domain),
                name=domain,
                base_url=base_url,
                robots_txt=None,
                crawl_delay=0.5,
                last_crawled=now,
                page_count=1,
                scraping_config={
                    "extraction_rules": extraction_rules or {},
                    "tool": preferred_tool,
                },
                description=website_description,
            )
            if scraping_job_created:
                await graph_db.add_node(websites_dict[base_url])
        else:
            websites_dict[base_url].page_count += 1
            # Update description for existing WebSite
            websites_dict[base_url].description = (
                f"Website: {domain}\n"
                f"Base URL: {base_url}\n"
                f"Last crawled: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Page count: {websites_dict[base_url].page_count}\n"
                f"Scraping tool: {preferred_tool}\n"
                f"Robots.txt: {'Available' if websites_dict[base_url].robots_txt else 'Not set'}\n"
                f"Crawl delay: {websites_dict[base_url].crawl_delay} seconds"
            )
            if scraping_job_created:
                await graph_db.add_node(websites_dict[base_url])

        # Create WebPage
        content_str = content if isinstance(content, str) else str(content)
        content_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        # Create description for WebPage
        webpage_description = (
            f"Webpage: {parsed_url.path.lstrip('/') or 'Home'}\n"
            f"URL: {page_url}\n"
            f"Scraped at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Content hash: {content_hash[:16]}...\n"
            f"Content type: text/html\n"
            f"Page size: {len(content_str)} bytes\n"
            f"Status code: 200"
        )

        webpage = WebPage(
            id=uuid5(NAMESPACE_OID, name=parsed_url.path.lstrip("/")),
            url=page_url,
            name=get_path_after_base(base_url, page_url),
            content=content_str,
            content_hash=content_hash,
            scraped_at=now,
            last_modified=None,
            status_code=200,
            content_type="text/html",
            page_size=len(content_str),
            extraction_rules=extraction_rules or {},
            description=webpage_description,
        )
        webpages.append(webpage)

    scraping_job.status = "completed" if webpages else "failed"
    # Update ScrapingJob description with final status
    scraping_job.description = (
        f"Scraping job: {job_name}\n"
        f"URLs: {', '.join(url)}\n"
        f"Status: {scraping_job.status}\n"
        f"Last run: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else 'Not scheduled'}"
    )

    websites = list(websites_dict.values())
    # Adding Nodes and Edges
    node_mapping = {scraping_job.id: scraping_job}
    edge_mapping = []

    for website in websites:
        node_mapping[website.id] = website
        edge_mapping.append(
            (
                scraping_job.id,
                website.id,
                "is_scraping",
                {
                    "source_node_id": scraping_job.id,
                    "target_node_id": website.id,
                    "relationship_name": "is_scraping",
                },
            )
        )

    for webpage in webpages:
        node_mapping[webpage.id] = webpage
        parsed_url = urlparse(webpage.url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        edge_mapping.append(
            (
                webpage.id,  # Corrected: WebPage is the source, WebSite is the target
                websites_dict[base_url].id,
                "is_part_of",
                {
                    "source_node_id": webpage.id,
                    "target_node_id": websites_dict[base_url].id,
                    "relationship_name": "is_part_of",
                },
            )
        )

    await graph_db.add_nodes(list(node_mapping.values()))
    await graph_db.add_edges(edge_mapping)
    await index_data_points(list(node_mapping.values()))
    await index_graph_edges()

    return await graph_db.get_graph_data()


def check_arguments(tavily_api_key, extraction_rules, tavily_config, soup_crawler_config):
    """
    Checking if the right argument are given, if not TypeError will be raised.
    """
    preferred_tool = "beautifulsoup"

    if extraction_rules and not soup_crawler_config:
        soup_crawler_config = SoupCrawlerConfig(extraction_rules=extraction_rules)

    if tavily_api_key:
        if not tavily_config:
            tavily_config = TavilyConfig(api_key=tavily_api_key)
        else:
            tavily_config.api_key = tavily_api_key
        if not extraction_rules and not soup_crawler_config:
            preferred_tool = "tavily"

    if not tavily_config and not soup_crawler_config:
        raise TypeError("Make sure you pass arguments for web_scraper_task")

    return soup_crawler_config, tavily_config, preferred_tool


def get_path_after_base(base_url, url):
    parsed_base = urlparse(base_url)
    parsed_url = urlparse(url)

    # Ensure they have the same netloc (domain)
    if parsed_base.netloc != parsed_url.netloc:
        raise ValueError("Base URL and target URL are from different domains")

    # Return everything after base_url path
    base_path = parsed_base.path.rstrip("/")
    full_path = parsed_url.path

    if full_path.startswith(base_path):
        return full_path[len(base_path) :].lstrip("/")
    else:
        return full_path.lstrip("/")
