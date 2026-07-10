import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4, UUID
from types import SimpleNamespace

import dlt
from cognee.tasks.ingestion.connectors.notion import notion
from cognee.tasks.ingestion.resolve_dlt_sources import resolve_dlt_sources, _delete_dlt_orphans
from cognee.tasks.ingestion.dlt_row_data import DltRowData
from cognee.modules.users.models import User

@pytest.mark.asyncio
async def test_notion_connector_fetching():
    """Test that the notion connector successfully invokes search and parses results."""
    mock_response_data = {
        "results": [
            {
                "id": "page-1",
                "object": "page",
                "last_edited_time": "2026-07-09T10:00:00.000Z",
                "url": "https://notion.so/page-1",
                "properties": {
                    "Name": {
                        "type": "title",
                        "title": [{"plain_text": "Notion Test Page"}]
                    }
                }
            }
        ],
        "has_more": False,
        "next_cursor": None
    }
    
    # Mock httpx.post inside connectors.notion
    with patch("httpx.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Instantiate notion source
        source = notion(api_key="fake_key")
        # Extract the page resource function
        fetch_pages_resource = source.resources["pages"]
        
        # Collect yielded pages by calling the underlying wrapped function directly
        pages = list(fetch_pages_resource.__wrapped__())
        
        assert len(pages) == 1
        assert pages[0]["id"] == "page-1"
        assert pages[0]["title"] == "Notion Test Page"
        assert pages[0]["last_edited_time"] == "2026-07-09T10:00:00.000Z"
        assert pages[0]["url"] == "https://notion.so/page-1"
        
        # Verify httpx.post was called with correct headers & url
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.notion.com/v1/search"
        assert kwargs["headers"]["Authorization"] == "Bearer fake_key"

@pytest.mark.asyncio
async def test_resolve_dlt_sources_enrichment():
    """Test that resolve_dlt_sources correctly injects the dlt_source_name into external_metadata."""
    # Create mock rows returned by DLT ingestion
    mock_row = DltRowData(
        table_name="pages",
        primary_key_column="id",
        primary_key_value="p-100",
        row_data={"title": "Test Row"},
        content_hash="hash123",
        schema_info=[],
        schema_hash="shash",
        foreign_keys=[],
        dlt_db_name="dlt_db",
        dataset_name="test_ds",
        dlt_source_name="notion" # Custom field populated by ingestion
    )
    
    # We mock ingest_dlt_source and get_unique_data_id to bypass actual database connection and return mock_row
    with patch("cognee.tasks.ingestion.resolve_dlt_sources.ingest_dlt_source", new_callable=AsyncMock) as mock_ingest, \
         patch("cognee.tasks.ingestion.resolve_dlt_sources.get_unique_data_id", new_callable=AsyncMock) as mock_get_id:
        mock_ingest.return_value = [mock_row]
        mock_get_id.return_value = uuid4()
        
        @dlt.source(name="notion")
        def mock_source():
            @dlt.resource(name="pages")
            def get_pages():
                yield {"id": "p-100", "title": "Test Row"}
            return get_pages
            
        mock_user = User(id=uuid4(), email="test@example.com")
        
        # Run resolve_dlt_sources
        result, cleanup_hook = await resolve_dlt_sources(
            data=mock_source(),
            dataset_name="test_ds",
            user=mock_user,
            write_disposition="merge"
        )
        
        assert len(result) == 1
        data_item = result[0]
        assert data_item.external_metadata["dlt_source_name"] == "notion"
        assert data_item.external_metadata["source"] == "dlt"

@pytest.mark.asyncio
async def test_orphan_cleanup_source_isolation():
    """Test that orphan cleanup of one DLT source does not delete rows from another source."""
    mock_user = User(id=uuid4(), email="test@example.com")
    
    # Mock datasets matching dataset_name
    mock_dataset = SimpleNamespace(id=uuid4())
    
    # Mock data items in the dataset
    notion_item = SimpleNamespace(
        id=uuid4(),
        external_metadata={"source": "dlt", "dlt_source_name": "notion"}
    )
    drive_item = SimpleNamespace(
        id=uuid4(),
        external_metadata={"source": "dlt", "dlt_source_name": "google_drive"}
    )
    
    with patch("cognee.modules.data.methods.get_authorized_existing_datasets", new_callable=AsyncMock) as mock_get_ds, \
         patch("cognee.modules.data.methods.get_dataset_data.get_dataset_data", new_callable=AsyncMock) as mock_get_data, \
         patch("cognee.modules.data.methods.delete_data.delete_data", new_callable=AsyncMock) as mock_delete, \
         patch("cognee.modules.graph.methods.has_data_related_nodes.has_data_related_nodes", new_callable=AsyncMock) as mock_has_nodes:
         
        mock_get_ds.return_value = [mock_dataset]
        mock_get_data.return_value = [notion_item, drive_item]
        mock_has_nodes.return_value = False
        
        # Simulate Notion sync run: fresh_data_ids doesn't contain notion_item (meaning it is orphaned),
        # but the active source set is {"notion"}.
        active_sources = {"notion"}
        fresh_ids = {drive_item.id} # drive_item is "fresh", notion_item is missing
        
        await _delete_dlt_orphans(
            dataset_name="test_ds",
            user=mock_user,
            fresh_data_ids=fresh_ids,
            active_dlt_source_names=active_sources
        )
        
        # delete_data should be called for notion_item (since it belongs to the active source "notion" and is missing)
        # but NOT for drive_item (which belongs to a different source and wasn't in active_dlt_source_names)
        mock_delete.assert_called_once_with(notion_item, mock_dataset.id)
