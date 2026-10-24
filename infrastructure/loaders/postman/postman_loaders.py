from typing import Any
import json

class PostmanLoader:
    """Loads Postman collections as text for cognee."""
    
    def _init_(self, path: str):
        self.path = path

    async def load(self) -> list[str]:
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            for item in data.get('item', []):
                name = item.get('name', '')
                request = item.get('request', {})
                url = request.get('url', {}).get('raw', '') if isinstance(request.get('url'), dict) else str(request.get('url', ''))
                method = request.get('method', '')
                desc = request.get('description', '')
                
                text = f"API: {name}\nMethod: {method}\nURL: {url}\nDescription: {desc}"
                texts.append(text)
            
            return texts
        except Exception as e:
            return [f"Error loading Postman collection: {str(e)}"]

def load_postman_collection(path: str) -> list[str]:
    import asyncio
    loader = PostmanLoader(path)
    return asyncio.run(loader.load())
