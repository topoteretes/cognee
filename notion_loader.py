import os
from typing import List, Dict, Any
from notion_client import Client

class NotionLoader:
    def __init__(self, auth_token: str = None):
        """
        Inițializează clientul Notion cu token-ul de integrare.
        """
        self.token = auth_token or os.getenv("NOTION_API_KEY")
        if not self.token:
            raise ValueError("Notion API Key este obligatorie (setează NOTION_API_KEY).")
        self.client = Client(auth=self.token)

    def fetch_page_content(self, page_id: str) -> str:
        """
        Extrage blocurile de text dintr-o pagină Notion.
        """
        try:
            blocks = self.client.blocks.children.list(block_id=page_id).get("results", [])
            text_content = []

            for block in blocks:
                block_type = block.get("type")
                if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item"]:
                    rich_texts = block.get(block_type, {}).get("rich_text", [])
                    plain_text = "".join([t.get("plain_text", "") for t in rich_texts])
                    if plain_text:
                        text_content.append(plain_text)

            return "\n".join(text_content)
        except Exception as e:
            print(f"Eroare la preluarea paginii {page_id}: {e}")
            return ""

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Caută toate paginile și bazele de date accesibile și returnează conținutul formatat.
        """
        response = self.client.search(query="")
        documents = []

        for result in response.get("results", []):
            if result.get("object") == "page":
                page_id = result["id"]
                title_list = result.get("properties", {}).get("title", {}).get("title", [])
                title = title_list[0]["plain_text"] if title_list else "Pagină Fără Titlu"
                
                content = self.fetch_page_content(page_id)
                if content:
                    documents.append({
                        "id": page_id,
                        "title": title,
                        "content": content
                    })
        return documents
