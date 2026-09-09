"""Regression test: GET /datasets/{id}/data is bounded, and counting does not fetch.

The route used to declare only ``dataset_id`` and ``user``, so FastAPI built its
request model without ``limit``/``offset`` and silently discarded them -- there
was no way for a caller to ask for fewer than every row. On a 171,828-document
dataset the endpoint answered in 408 s with 70 MB, which is what made the
dataset page spin forever.

Two things are asserted here: the route now pages (with a bounded default, so an
un-updated client benefits without changing), and ``/data/count`` answers the
"how many documents" question without serializing any rows.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

DATASET_ID = uuid.uuid4()
ROW_COUNT = 250


def _row(index: int) -> SimpleNamespace:
    """A stand-in for a Data ORM row, carrying only what DataDTO reads."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        name=f"doc-{index}.txt",
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        extension="txt",
        mime_type="text/plain",
        raw_data_location=f"/tmp/doc-{index}.txt",
        # Nullable in the model: the route must substitute the requested id
        # rather than validate this through.
        dataset_id=None,
        label=None,
        external_metadata=None,
    )


ROWS = [_row(i) for i in range(ROW_COUNT)]


@pytest.fixture
def client(monkeypatch):
    import importlib

    module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(module, "send_telemetry", lambda *a, **k: None)

    async def _authorized(dataset_ids, permission, user):
        return [SimpleNamespace(id=DATASET_ID)]

    monkeypatch.setattr(module, "get_authorized_existing_datasets", _authorized)

    methods = importlib.import_module("cognee.modules.data.methods")

    async def _get_dataset_data(dataset_id, limit=None, offset=0):
        window = ROWS[offset:]
        return window[:limit] if limit is not None else window

    async def _count_dataset_data(dataset_id):
        return len(ROWS)

    monkeypatch.setattr(methods, "get_dataset_data", _get_dataset_data)
    monkeypatch.setattr(methods, "count_dataset_data", _count_dataset_data)

    app = FastAPI()
    app.include_router(module.get_datasets_router(), prefix="/api/v1/datasets")

    async def _user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    app.dependency_overrides[get_authenticated_user] = _user
    with TestClient(app) as c:
        yield c


def test_default_response_is_bounded(client):
    """No query parameters must not mean "every row"."""
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 100, "the default page must be bounded, not the whole dataset"
    assert len(body) < ROW_COUNT


def test_limit_and_offset_page_through_without_gaps_or_repeats(client):
    first = client.get(f"/api/v1/datasets/{DATASET_ID}/data?limit=100&offset=0").json()
    second = client.get(f"/api/v1/datasets/{DATASET_ID}/data?limit=100&offset=100").json()
    tail = client.get(f"/api/v1/datasets/{DATASET_ID}/data?limit=100&offset=200").json()

    assert [len(first), len(second), len(tail)] == [100, 100, 50]

    seen = [item["id"] for item in first + second + tail]
    assert len(set(seen)) == ROW_COUNT, "paging must not repeat or skip rows"


@pytest.mark.parametrize("query", ["limit=0", "limit=1001", "limit=-1", "offset=-1"])
def test_out_of_range_paging_is_rejected(client, query):
    """Rejected loudly, not silently clamped.

    The code is 422 here and 400 in the real app: cognee registers its own
    request_validation_exception_handler, which this bare test app does not
    mount. What matters either way is that the request is refused rather than
    quietly answered with a clamped page.
    """
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?{query}")

    assert response.status_code in (400, 422)


def test_requested_dataset_id_wins_over_the_nullable_column(client):
    """Rows carry a nullable dataset_id; the DTO's is required, so the route substitutes."""
    body = client.get(f"/api/v1/datasets/{DATASET_ID}/data?limit=1").json()

    assert body[0]["datasetId"] == str(DATASET_ID)


def test_count_endpoint_returns_the_total(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data/count")

    assert response.status_code == 200
    assert response.json() == {"count": ROW_COUNT}


def test_count_segment_is_not_shadowed_by_a_parameter_route():
    """A GET /{dataset_id}/data/{param} registered first would swallow /data/count.

    Asserted against the router's own route table rather than a response code:
    a handler can 404 for reasons of its own, which would let a status-code
    check agree for the wrong reason.
    """
    import re

    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    get_paths = [
        route.path for route in get_datasets_router().routes if "GET" in (route.methods or set())
    ]

    assert "/{dataset_id}/data/count" in get_paths

    # Any GET sibling of the same depth whose third segment is a path parameter
    # would match "count" first if it were registered earlier.
    sibling = re.compile(r"^/\{dataset_id\}/data/\{[^}]+\}$")
    shadowing = [path for path in get_paths if sibling.match(path)]

    assert not shadowing, f"these GET routes would swallow /data/count: {shadowing}"
