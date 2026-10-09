import httpx
from typing import List, Dict, Any

async def read_raindrop_data(token: str) -> List[Dict[str, Any]]:
    """
    Fetches saved bookmarks/raindrops from Raindrop.io API to ingest into Cognee memory.
    """
    url = "https://api.raindrop.io/rest/v1/raindrops/0"
    headers = {"Authorization": f"Bearer {token}"}
    documents = []

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            for item in data.get("items", []):
                title = item.get("title", "")
                link = item.get("link", "")
                excerpt = item.get("excerpt", "")
                tags = ", ".join(item.get("tags", []))

                text_content = f"Title: {title}\nURL: {link}\nExcerpt: {excerpt}\nTags: {tags}"

                documents.append({
                    "id": str(item.get("_id")),
                    "text": text_content,
                    "metadata": {
                        "source": "raindrop",
                        "domain": item.get("domain", ""),
                        "created": item.get("created", "")
                    }
                })

            return documents

        except Exception as e:
            print(f"Error fetching data from Raindrop API: {e}")
            return []