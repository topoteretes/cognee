import aiohttp
from typing import List, Dict, Any, AsyncGenerator

async def fetch_pocket_items(consumer_key: str, access_token: str, state: str = "unread") -> List[Dict[str, Any]]:
    url = "https://getpocket.com/v3/get"
    payload = {
        "consumer_key": consumer_key,
        "access_token": access_token,
        "state": state,
        "detailType": "complete"
    }
    headers = {"Content-Type": "application/json; charset=UTF-8", "X-Accept": "application/json"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                list_data = data.get("list", {})
                if isinstance(list_data, dict):
                    return list(list_data.values())
                return list_data
            return []

async def read_pocket(consumer_key: str, access_token: str) -> AsyncGenerator[Dict[str, Any], None]:
    items = await fetch_pocket_items(consumer_key, access_token)
    for item in items:
        yield {
            "id": item.get("item_id"),
            "resolved_id": item.get("resolved_id"),
            "given_title": item.get("given_title", ""),
            "resolved_title": item.get("resolved_title", ""),
            "url": item.get("given_url") or item.get("resolved_url", ""),
            "excerpt": item.get("excerpt", ""),
            "time_added": item.get("time_added", ""),
        }