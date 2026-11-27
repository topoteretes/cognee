import aiohttp
import feedparser
from typing import List, Dict, Any, AsyncGenerator

async def read_substack_feed(substack_url: str) -> AsyncGenerator[Dict[str, Any], None]:
    # Ensure RSS URL format
    if not substack_url.endswith("/feed"):
        rss_url = substack_url.rstrip("/") + "/feed"
    else:
        rss_url = substack_url

    async with aiohttp.ClientSession() as session:
        async with session.get(rss_url) as response:
            if response.status == 200:
                content = await response.text()
                feed = feedparser.parse(content)
                
                for entry in feed.entries:
                    yield {
                        "id": getattr(entry, "id", getattr(entry, "link", "")),
                        "title": getattr(entry, "title", ""),
                        "link": getattr(entry, "link", ""),
                        "published": getattr(entry, "published", ""),
                        "summary": getattr(entry, "summary", ""),
                        "author": getattr(entry, "author", ""),
                    }