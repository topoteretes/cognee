import os
from typing import Optional
import httpx

try:
    import dlt
except ImportError:
    dlt = None

def notion(
    api_key: Optional[str] = None,
    database_id: Optional[str] = None,
):
    """Notion SaaS source connector built on Cognee DLT ingestion.
    
    Pulls pages from the Notion workspace, supporting incremental loading
    based on page last_edited_time, and primary key mapping for merge deduplication.
    """
    if dlt is None:
        raise ImportError(
            "The 'dlt' package is required to use SaaS source connectors. "
            "Please install it using 'pip install dlt'."
        )
    
    api_key = api_key or os.getenv("NOTION_API_KEY")
    if not api_key:
        raise ValueError(
            "Notion API Key is required. Pass api_key or set the NOTION_API_KEY environment variable."
        )

    @dlt.source(name="notion")
    def notion_source():
        @dlt.resource(
            name="pages",
            write_disposition="merge",
            primary_key="id",
        )
        def fetch_pages(
            last_edited_time = dlt.sources.incremental(
                "last_edited_time",
                initial_value="1970-01-01T00:00:00.000Z"
            )
        ):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            }
            
            url = "https://api.notion.com/v1/search"
            payload = {
                "filter": {
                    "value": "page",
                    "property": "object"
                },
                "sort": {
                    "direction": "ascending",
                    "timestamp": "last_edited_time"
                }
            }
            
            start_cursor = None
            
            while True:
                if start_cursor:
                    payload["start_cursor"] = start_cursor
                
                response = httpx.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                for result in data.get("results", []):
                    # Extract the text content from the page properties for indexing
                    title = ""
                    properties = result.get("properties", {})
                    for prop_name, prop_val in properties.items():
                        if prop_val.get("type") == "title":
                            title_parts = prop_val.get("title", [])
                            title = "".join(part.get("plain_text", "") for part in title_parts)
                            break
                    
                    page_id = result.get("id")
                    last_edited = result.get("last_edited_time")
                    
                    # Yield structured dictionary
                    yield {
                        "id": page_id,
                        "title": title,
                        "last_edited_time": last_edited,
                        "url": result.get("url"),
                        "raw_properties": properties,
                    }
                    
                if not data.get("has_more"):
                    break
                start_cursor = data.get("next_cursor")
                
        return fetch_pages
        
    return notion_source()
