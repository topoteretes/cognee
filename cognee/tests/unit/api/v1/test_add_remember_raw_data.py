"""The add and remember endpoints accept string inputs through `raw_data`.

Each entry is one data item — raw text, a server-side path, a web URL, or a
GitHub/GitLab repository URL — and joins the uploads as a single list, uploads
first. The router does no interpretation of its own: the strings reach add()
exactly as an SDK caller would pass them, and labels/external_metadata pair
with the combined list. Empty entries (Swagger UI submits untouched array items
as "") are dropped. A request with neither uploads nor raw_data is a 400, and
remember rejects raw_data together with any content_type because those paths
never run string inputs through add().
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.exceptions import LabelCountMismatchError

import importlib

add_pkg = importlib.import_module("cognee.api.v1.add")
remember_pkg = importlib.import_module("cognee.api.v1.remember")

REPO_URL = "https://github.com/org/repo"
UPLOAD = ("data", ("first.txt", b"first file", "text/plain"))
MOCK_USER = SimpleNamespace(
    id=str(uuid.uuid4()), email="test@example.com", is_active=True, tenant_id=str(uuid.uuid4())
)


@pytest.fixture
def client():
    """Just the two routers: no app lifespan, so no databases are touched. Router
    errors raised as CogneeApiError surface as exceptions (the global handler
    that turns them into 400s lives on the full app)."""
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/api/v1/add")
    app.include_router(get_remember_router(), prefix="/api/v1/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


def pipeline_run_completed():
    return PipelineRunCompleted(
        pipeline_run_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        dataset_name="test_dataset",
    )


def remember_completed():
    return SimpleNamespace(status="completed", to_dict=lambda: {"status": "completed"})


# ── add ─────────────────────────────────────────────────────────────────────


def test_add_forwards_raw_data_strings_untouched(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            data={
                "datasetName": "test_dataset",
                "raw_data": ["Some text to remember", REPO_URL, "/srv/docs/report.pdf"],
            },
        )

        assert response.status_code == 200, response.text
        assert mock_add.call_args.args[0] == [
            "Some text to remember",
            REPO_URL,
            "/srv/docs/report.pdf",
        ]


def test_add_puts_uploads_before_raw_data_and_pairs_labels_in_that_order(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=[UPLOAD],
            data={
                "datasetName": "test_dataset",
                "raw_data": [REPO_URL],
                "labels": '["docs", "code"]',
            },
        )

        assert response.status_code == 200, response.text
        sent = mock_add.call_args.args[0]
        assert [type(item) for item in sent] == [DataItem, DataItem]
        assert sent[0].data.filename == "first.txt"
        assert sent[1].data == REPO_URL
        assert [item.label for item in sent] == ["docs", "code"]


def test_add_drops_empty_raw_data_entries_and_strips_whitespace(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            data={"datasetName": "test_dataset", "raw_data": ["", "  ", f"  {REPO_URL}  "]},
        )

        assert response.status_code == 200, response.text
        assert mock_add.call_args.args[0] == [REPO_URL]


@pytest.mark.parametrize(
    "form",
    [{}, {"raw_data": ""}, {"raw_data": ["", " "]}],
    ids=["nothing", "blank_field", "only_empty_entries"],
)
def test_add_requires_data_or_raw_data(client, form):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post("/api/v1/add", data={"datasetName": "test_dataset", **form})

        assert response.status_code == 400
        assert "raw_data" in response.json()["error"]
        mock_add.assert_not_awaited()


def test_add_label_count_is_checked_against_the_combined_list(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        with pytest.raises(LabelCountMismatchError, match="got 1 labels for 2 items"):
            client.post(
                "/api/v1/add",
                files=[UPLOAD],
                data={"datasetName": "test_dataset", "raw_data": [REPO_URL], "labels": '["docs"]'},
            )

        mock_add.assert_not_awaited()


# ── remember ────────────────────────────────────────────────────────────────


def test_remember_forwards_raw_data_strings_untouched(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember",
            data={"datasetName": "test_dataset", "raw_data": ["Some text", REPO_URL]},
        )

        assert response.status_code == 200, response.text
        assert mock_remember.call_args.args[0] == ["Some text", REPO_URL]


def test_remember_raw_data_works_with_session_id(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember",
            data={
                "datasetName": "test_dataset",
                "session_id": "claude-code-1718000000",
                "raw_data": ["User prefers short answers"],
            },
        )

        assert response.status_code == 200, response.text
        assert mock_remember.call_args.args[0] == ["User prefers short answers"]
        assert mock_remember.call_args.kwargs["session_id"] == "claude-code-1718000000"


def test_remember_combines_uploads_and_raw_data_uploads_first(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember",
            files=[UPLOAD],
            data={"datasetName": "test_dataset", "raw_data": [REPO_URL]},
        )

        assert response.status_code == 200, response.text
        sent = mock_remember.call_args.args[0]
        assert sent[0].filename == "first.txt"
        assert sent[1] == REPO_URL


@pytest.mark.parametrize("content_type", ["code", "skills", "cogx-archive"])
def test_remember_rejects_raw_data_with_content_type(client, content_type):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            data={
                "datasetName": "test_dataset",
                "content_type": content_type,
                "raw_data": [REPO_URL],
            },
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "raw_data" in detail and "repositories" in detail
        mock_remember.assert_not_awaited()


def test_remember_requires_data_or_raw_data_for_normal_ingestion(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post("/api/v1/remember", data={"datasetName": "test_dataset"})

        assert response.status_code == 400
        assert "raw_data" in response.json()["detail"]
        mock_remember.assert_not_awaited()
