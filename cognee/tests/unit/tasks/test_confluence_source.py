import pytest
from unittest.mock import patch, MagicMock
import dlt
from cognee.tasks.ingestion.confluence_source import create_confluence_source, get_all_current_page_ids
from cognee.tasks.ingestion.create_dlt_source import is_confluence_config

@patch("requests.Session")
def test_pages_resource_calls_correct_api(mock_session):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"id": "1", "version": {"when": "2024-01-01"}}]}
    mock_session.return_value.get.return_value = mock_resp
    
    wrapper = create_confluence_source("https://test.net", "u@b.c", "tok", ["ENG"])
    pages_resource = wrapper.source.resources["pages"]
    
    items = list(pages_resource)
    assert len(items) == 1
    assert items[0]["id"] == "1"
    
    # Verify API hit
    mock_session.return_value.get.assert_called_with(
        "https://test.net/wiki/api/v2/spaces/ENG/pages",
        params={'limit': 250, 'sort': 'modified-date', 'body-format': 'storage'}
    )

@patch("requests.Session")
def test_incremental_sync_pipeline_behavior(mock_session):
    """Proves the incremental cursor persists and filters across syncs using a real DLT pipeline."""
    # First sync returns two pages
    mock_resp_1 = MagicMock()
    mock_resp_1.json.return_value = {
        "results": [
            {"id": "1", "version": {"when": "2024-01-01T10:00:00.000Z"}},
            {"id": "2", "version": {"when": "2024-01-02T10:00:00.000Z"}}
        ]
    }
    
    # Second sync returns the same two pages (simulating unchanged upstream)
    mock_resp_2 = MagicMock()
    mock_resp_2.json.return_value = {
        "results": [
            {"id": "1", "version": {"when": "2024-01-01T10:00:00.000Z"}},
            {"id": "2", "version": {"when": "2024-01-02T10:00:00.000Z"}}
        ]
    }
    
    mock_session.return_value.get.side_effect = [mock_resp_1, mock_resp_2]
    
    pipeline = dlt.pipeline(pipeline_name="test_confluence", destination="duckdb", dataset_name="test_ds")
    
    wrapper = create_confluence_source("https://test.net", "u@b.c", "tok", ["ENG"])
    info1 = pipeline.run(wrapper.source.with_resources("pages"))
    
    assert info1.has_failed_jobs is False
    with pipeline.sql_client() as client:
        res = client.execute_sql("SELECT count(*) FROM pages")
        assert res[0][0] == 2
        
    wrapper2 = create_confluence_source("https://test.net", "u@b.c", "tok", ["ENG"])
    info2 = pipeline.run(wrapper2.source.with_resources("pages"))
    
    assert info2.has_failed_jobs is False
    # Second run should extract 0 rows because cursor prevents yielding new jobs
    assert len(info2.load_packages) == 0 or all(not pkg.jobs.get("new_jobs") for pkg in info2.load_packages)
    
    with pipeline.sql_client() as client:
        # Data tables remain at 2
        res = client.execute_sql("SELECT count(*) FROM pages")
        assert res[0][0] == 2

@patch("requests.Session")
def test_full_id_sweep(mock_session):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"id": "123"}, {"id": "456"}]}
    mock_session.return_value.get.return_value = mock_resp
    
    active_ids = get_all_current_page_ids("https://test.net", "u@b.c", "tok", ["ENG"])
    assert active_ids == {"123", "456"}
    
    # Assert it uses the lightweight select=id param
    mock_session.return_value.get.assert_called_with(
        "https://test.net/wiki/api/v2/spaces/ENG/pages",
        params={'limit': 250, 'select': 'id'}
    )

def test_is_confluence_config_valid():
    assert is_confluence_config({
        "confluence_url": "https://x.atlassian.net",
        "email": "a@b.com",
        "api_token": "tok",
    })

def test_is_confluence_config_invalid():
    assert not is_confluence_config({"url": "https://..."})
    assert not is_confluence_config("not a dict")
