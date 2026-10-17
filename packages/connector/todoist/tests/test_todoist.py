"""Comprehensive unit tests for the Todoist connector.

All Todoist API calls are mocked — no network access or API token is required.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee_community_connector_todoist.todoist import (
    TodoistConnector,
    TodoistConnectorError,
)

# ---------------------------------------------------------------------------
# Fixtures — representative JSON from the Todoist REST API v2
# ---------------------------------------------------------------------------

MOCK_PROJECTS: List[Dict[str, Any]] = [
    {
        "id": "2203306141",
        "name": "Inbox",
        "comment_count": 0,
        "order": 0,
        "color": "grey",
        "is_shared": False,
        "is_favorite": False,
        "is_inbox_project": True,
        "is_team_inbox": False,
        "view_style": "list",
        "url": "https://todoist.com/showProject?id=2203306141",
    },
    {
        "id": "2203306142",
        "name": "Work",
        "comment_count": 3,
        "order": 1,
        "color": "blue",
        "is_shared": False,
        "is_favorite": True,
        "is_inbox_project": False,
        "is_team_inbox": False,
        "view_style": "board",
        "url": "https://todoist.com/showProject?id=2203306142",
    },
]

MOCK_TASKS: List[Dict[str, Any]] = [
    {
        "id": "8001",
        "project_id": "2203306141",
        "content": "Buy groceries",
        "description": "Milk, eggs, bread",
        "priority": 4,
        "labels": ["errands", "shopping"],
        "due": {"date": "2025-12-01", "is_recurring": False, "string": "Dec 1"},
        "url": "https://todoist.com/showTask?id=8001",
    },
    {
        "id": "8002",
        "project_id": "2203306142",
        "content": "Write quarterly report",
        "description": "",
        "priority": 3,
        "labels": [],
        "due": None,
        "url": "https://todoist.com/showTask?id=8002",
    },
    {
        # Minimal task — missing optional fields
        "id": "8003",
        "project_id": "2203306142",
        "content": "Quick follow-up",
        "priority": 1,
    },
]


# ---------------------------------------------------------------------------
# Helper to build a connector with mocked HTTP
# ---------------------------------------------------------------------------


def _mock_get(projects: Any = None, tasks: Any = None) -> MagicMock:
    """Return a patched ``httpx.get`` that serves canned responses."""
    if projects is None:
        projects = MOCK_PROJECTS
    if tasks is None:
        tasks = MOCK_TASKS

    def side_effect(url: str, **kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        if "/projects" in url:
            resp.json.return_value = projects
        elif "/tasks" in url:
            resp.json.return_value = tasks
        else:
            resp.status_code = 404
            resp.text = "Not Found"
        return resp

    return MagicMock(side_effect=side_effect)


# ---------------------------------------------------------------------------
# Tests — initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_valid_token(self) -> None:
        connector = TodoistConnector(api_token="test-token-123")
        assert connector._api_token == "test-token-123"
        assert "Bearer test-token-123" in connector._headers["Authorization"]

    def test_empty_token_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            TodoistConnector(api_token="")


# ---------------------------------------------------------------------------
# Tests — fetch_projects
# ---------------------------------------------------------------------------


class TestFetchProjects:
    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_returns_projects(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        projects = connector.fetch_projects()
        assert len(projects) == 2
        assert projects[0]["name"] == "Inbox"
        assert projects[1]["name"] == "Work"

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_auth_header_sent(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="secret-tok")
        connector.fetch_projects()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret-tok"

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_api_error_raises(self, mock_get: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 403
        resp.text = "Forbidden"
        mock_get.return_value = resp

        connector = TodoistConnector(api_token="bad-tok")
        with pytest.raises(TodoistConnectorError, match="403"):
            connector.fetch_projects()


# ---------------------------------------------------------------------------
# Tests — fetch_tasks
# ---------------------------------------------------------------------------


class TestFetchTasks:
    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_returns_all_tasks(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        tasks = connector.fetch_tasks()
        assert len(tasks) == 3

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_project_filter_passes_param(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        connector.fetch_tasks(project_id="2203306141")
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {"project_id": "2203306141"}

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_no_filter_sends_no_params(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        connector.fetch_tasks()
        _, kwargs = mock_get.call_args
        assert kwargs["params"] is None


# ---------------------------------------------------------------------------
# Tests — extract_task_documents
# ---------------------------------------------------------------------------


class TestExtractTaskDocuments:
    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_document_count_matches_tasks(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert len(docs) == 3

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_project_name_resolved(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "**Project:** Inbox" in docs[0]
        assert "**Project:** Work" in docs[1]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_priority_mapping(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        # Task 0: API priority 4 → P1 (Urgent)
        assert "P1 (Urgent)" in docs[0]
        # Task 1: API priority 3 → P2 (Medium)
        assert "P2 (Medium)" in docs[1]
        # Task 2: API priority 1 → P4 (No priority)
        assert "P4 (No priority)" in docs[2]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_due_date_present(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "2025-12-01" in docs[0]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_due_date_none_handled(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        # Task 1 has due=None, task 2 has no due field at all
        assert "No due date" in docs[1]
        assert "No due date" in docs[2]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_description_included(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "Milk, eggs, bread" in docs[0]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_empty_description_omitted(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        # Task 1 has empty description — section header should be absent
        assert "## Description" not in docs[1]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_labels_rendered(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "errands, shopping" in docs[0]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_empty_labels_shows_none(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "**Labels:** None" in docs[1]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_missing_optional_fields_no_crash(
        self, mock_get: MagicMock
    ) -> None:
        """Task 8003 lacks description, labels, due, and url fields."""
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "Quick follow-up" in docs[2]
        assert "**Labels:** None" in docs[2]
        assert "No due date" in docs[2]

    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    def test_task_url_included(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")
        docs = connector.extract_task_documents()
        assert "[Open in Todoist]" in docs[0]


# ---------------------------------------------------------------------------
# Tests — ingest_tasks (async)
# ---------------------------------------------------------------------------


class TestIngestTasks:
    @pytest.mark.asyncio
    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    async def test_ingest_calls_add_and_cognify(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")

        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        count = await connector.ingest_tasks(mock_cognee)
        assert count == 3
        assert mock_cognee.add.call_count == 3
        mock_cognee.cognify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    async def test_ingest_returns_zero_when_no_tasks(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.side_effect = _mock_get(tasks=[]).side_effect
        connector = TodoistConnector(api_token="tok")

        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        count = await connector.ingest_tasks(mock_cognee)
        assert count == 0
        mock_cognee.add.assert_not_called()
        mock_cognee.cognify.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("cognee_community_connector_todoist.todoist.httpx.get")
    async def test_ingest_with_project_filter(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.side_effect = _mock_get().side_effect
        connector = TodoistConnector(api_token="tok")

        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        count = await connector.ingest_tasks(
            mock_cognee, project_id="2203306141"
        )
        assert count == 3
        mock_cognee.cognify.assert_awaited_once()
