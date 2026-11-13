import aiohttp
from typing import List, Dict, Any, AsyncGenerator

async def fetch_gdrive_files(access_token: str, query: str = "trashed = false") -> List[Dict[str, Any]]:
    url = "https://www.googleapis.com/drive/v3/files"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    params = {
        "q": query,
        "fields": "files(id, name, mimeType, webViewLink, createdTime, modifiedTime)"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("files", [])
            return []

async def read_gdrive(access_token: str) -> AsyncGenerator[Dict[str, Any], None]:
    files = await fetch_gdrive_files(access_token)
    for file_item in files:
        yield {
            "id": file_item.get("id"),
            "name": file_item.get("name", ""),
            "mime_type": file_item.get("mimeType", ""),
            "web_view_link": file_item.get("webViewLink", ""),
            "created_time": file_item.get("createdTime", ""),
            "modified_time": file_item.get("modifiedTime", "")
        }