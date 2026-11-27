import aiohttp
from typing import List, Dict, Any, AsyncGenerator

async def fetch_hacker_news_story(story_id: int, session: aiohttp.ClientSession) -> Dict[str, Any]:
    url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        return {}

async def read_hacker_news(story_ids: List[int]) -> AsyncGenerator[Dict[str, Any], None]:
    async with aiohttp.ClientSession() as session:
        for story_id in story_ids:
            story_data = await fetch_hacker_news_story(story_id, session)
            if story_data and story_data.get("type") == "story":
                yield {
                    "id": story_data.get("id"),
                    "title": story_data.get("title", ""),
                    "url": story_data.get("url", ""),
                    "by": story_data.get("by", ""),
                    "score": story_data.get("score", 0),
                    "time": story_data.get("time", 0),
                    "text": story_data.get("text", ""),
                }