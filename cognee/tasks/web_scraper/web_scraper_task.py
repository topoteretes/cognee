from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges
from cognee.infrastructure.databases.graph import get_graph_engine
from .models import WebPage, WebSite, ScrapingJob
from typing import Union, List, Dict
from urllib.parse import urlparse


async def web_scraper_task(url: Union[str, List[str]], **kwargs):
    graph_engine = await get_graph_engine()
    # Mapping between parsed_url object and urls
    mappings = {}
    web_scraping_job = ScrapingJob(
        job_name="default_job",
        urls=[url] if isinstance(url, str) else url,
        scraping_rules={},
        schedule=None,
        status="active",
        last_run=None,
        next_run=None,
    )
    data_point_mappings: Dict[WebSite, List[WebPage]] = {}
    if isinstance(url, List):
        for single_url in url:
            parsed_url = urlparse(single_url)
            domain = parsed_url.netloc
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            if mappings.get(parsed_url):
                mappings[parsed_url] = [single_url]
            else:
                mappings[parsed_url].append(single_url)
    else:
        if mappings.get(parsed_url):
            mappings[parsed_url] = [single_url]
        else:
            mappings[parsed_url].append(single_url)
    for parsed_url in mappings.keys():
        domain = parsed_url.netloc
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        web_site = WebSite(
            domain=domain,
            base_url=base_url,
            robots_txt="",
            crawl_delay=0,
            last_crawled=None,
            page_count=0,
            scraping_config={},
        )
        for url in mappings[parsed_url]:
            # Process each URL with the web scraping logic
            web_page = WebPage(
                url=url,
                title="",
                content="",
                content_hash="",
                scraped_at=None,
                last_modified=None,
                status_code=0,
                content_type="",
                page_size=0,
                extraction_rules={},
            )
