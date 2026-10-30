import pytest
from unittest.mock import MagicMock, patch
from notion_loader import NotionLoader

@patch("notion_loader.Client")
def test_notion_loader_init(mock_client):
    loader = NotionLoader(auth_token="test_token")
    assert loader.token == "test_token"

@patch("notion_loader.Client")
def test_fetch_page_content(mock_client):
    mock_instance = mock_client.return_value
    mock_instance.blocks.children.list.return_value = {
        "results": [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"plain_text": "Hello Cognee!"}]
                }
            }
        ]
    }
    
    loader = NotionLoader(auth_token="test_token")
    content = loader.fetch_page_content("fake_page_id")
    assert content == "Hello Cognee!"