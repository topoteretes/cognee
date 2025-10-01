from tavily import AsyncTavilyClient
from bs4 import BeautifulSoup
import os
import requests
from typing import Dict, Any, List, Union


async def fetch_page_content(urls: Union[str, List[str]], extraction_rules: Dict[str, Any]) -> str:
    if os.getenv("TAVILY_API_KEY") is not None:
        return await fetch_with_tavily(urls)
    else:
        return await fetch_with_bs4(urls, extraction_rules)


async def fetch_with_tavily(urls: Union[str, List[str]]) -> Dict[str, str]:
    client = AsyncTavilyClient()
    results = await client.extract(urls, include_images=False)
    result_dict = {}
    for result in results["results"]:
        result_dict[result["url"]] = result["raw_content"]
    return result_dict


async def fetch_with_bs4(urls: Union[str,List[str]], extraction_rules: Dict) -> Dict[str]:
    result_dict = {}
    if isinstance(urls,str):
        urls = [urls]
    for url in urls:
        response = requests.get(url, headers={"User-Agent": "Cognee-Scraper"})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        extracted_data = ""

        for field, selector in extraction_rules.items():
            element = soup.select_one(selector)
            extracted_data += element.get_text(strip=True) if element else ""

    return result_dict
