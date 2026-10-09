"""GET /datasets/graph-summary: the poll-friendly counts endpoint.

The endpoint is meant to be polled, so a transient relational failure has to
come back the way its siblings' do — a 409 with a generic message — instead of
an unhandled 500 that puts the exception text in front of the caller. These
pin the two shapes a client sees: the summary itself, and that failure.
"""

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.data.methods import DatasetGraphCounts
from cognee.modules.users.methods import get_authenticated_user

router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")

DATASET_ID = uuid4()
RUN_ID = uuid4()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=DATASET_ID, name="billing")]),
    )

    app = FastAPI()
    app.include_router(router_module.get_datasets_router(), prefix="/api/v1/datasets")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), email="user@example.com", is_active=True, tenant_id=None
    )
    return TestClient(app)


def test_a_dataset_is_summarized_with_its_cached_counts(client, monkeypatch):
    cached_at = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        router_module,
        "get_datasets_graph_counts",
        AsyncMock(
            return_value={
                DATASET_ID: DatasetGraphCounts(
                    pipeline_run_id=RUN_ID, num_nodes=12, num_edges=34, computed_at=cached_at
                )
            }
        ),
    )

    response = client.get(f"/api/v1/datasets/graph-summary?dataset_ids={DATASET_ID}")

    assert response.status_code == 200
    assert response.json() == [
        {
            "datasetId": str(DATASET_ID),
            "pipelineRunId": str(RUN_ID),
            "numNodes": 12,
            "numEdges": 34,
            # The API's own UTC encoding, not datetime.isoformat()'s "+00:00".
            "computedAt": "2026-08-03T09:00:00Z",
        }
    ]


def test_no_readable_datasets_is_an_empty_list_not_an_error(client, monkeypatch):
    monkeypatch.setattr(
        router_module, "get_authorized_existing_datasets", AsyncMock(return_value=[])
    )

    response = client.get("/api/v1/datasets/graph-summary")

    assert response.status_code == 200
    assert response.json() == []


def test_a_failed_count_read_is_a_409_that_leaks_nothing(client, monkeypatch):
    """A transient DB fault on a polled endpoint must not surface as a 500
    carrying the driver's error text."""
    monkeypatch.setattr(
        router_module,
        "get_datasets_graph_counts",
        AsyncMock(side_effect=RuntimeError("connection pool exhausted at 10.0.0.4:5432")),
    )

    response = client.get(f"/api/v1/datasets/graph-summary?dataset_ids={DATASET_ID}")

    assert response.status_code == 409
    assert response.json() == {"error": "Unable to retrieve dataset graph summary."}
    assert "10.0.0.4" not in response.text


def test_a_failed_authorization_read_is_also_a_409_not_a_500(client, monkeypatch):
    """The authorization lookup is the same kind of relational read as the
    counts lookup — a transient fault there must get the same 409, not an
    unhandled 500."""
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(side_effect=RuntimeError("connection pool exhausted at 10.0.0.4:5432")),
    )

    response = client.get(f"/api/v1/datasets/graph-summary?dataset_ids={DATASET_ID}")

    assert response.status_code == 409
    assert response.json() == {"error": "Unable to retrieve dataset graph summary."}
    assert "10.0.0.4" not in response.text
