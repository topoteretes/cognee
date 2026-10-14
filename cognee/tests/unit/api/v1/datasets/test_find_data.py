"""``datasets.find_data`` and the ``content_hash`` filter on GET /datasets/{id}/data."""

import importlib
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.users.methods import get_authenticated_user

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

DATASET_ID = uuid.uuid4()
CONTENT = "the text that was added"
CONTENT_HASH = compute_content_hash(CONTENT)


# --- SDK: datasets.find_data -------------------------------------------------


@pytest.mark.asyncio
async def test_find_data_hashes_content_and_looks_up_by_hash():
    user = SimpleNamespace(id=uuid.uuid4())
    dataset = SimpleNamespace(id=DATASET_ID)
    row = SimpleNamespace(id=uuid.uuid4(), content_hash=CONTENT_HASH)
    lookup = AsyncMock(return_value=[row])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
    ):
        found = await datasets_module.datasets.find_data(DATASET_ID, content=CONTENT, user=user)

    assert found == [row]
    lookup.assert_awaited_once_with(DATASET_ID, CONTENT_HASH)


@pytest.mark.asyncio
async def test_find_data_accepts_a_precomputed_hash():
    user = SimpleNamespace(id=uuid.uuid4())
    dataset = SimpleNamespace(id=DATASET_ID)
    lookup = AsyncMock(return_value=[])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
    ):
        found = await datasets_module.datasets.find_data(
            DATASET_ID, content_hash=CONTENT_HASH, user=user
        )

    assert found == []
    lookup.assert_awaited_once_with(DATASET_ID, CONTENT_HASH)


@pytest.mark.asyncio
async def test_find_data_requires_exactly_one_selector():
    user = SimpleNamespace(id=uuid.uuid4())
    with pytest.raises(ValueError):
        await datasets_module.datasets.find_data(DATASET_ID, user=user)
    with pytest.raises(ValueError):
        await datasets_module.datasets.find_data(
            DATASET_ID, content=CONTENT, content_hash=CONTENT_HASH, user=user
        )


@pytest.mark.asyncio
async def test_find_data_checks_read_access_before_looking_up():
    user = SimpleNamespace(id=uuid.uuid4())
    authorize = AsyncMock(side_effect=PermissionError("no access"))
    lookup = AsyncMock()

    with (
        patch.object(datasets_module, "get_authorized_dataset", authorize),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
        pytest.raises(PermissionError),
    ):
        await datasets_module.datasets.find_data(DATASET_ID, content=CONTENT, user=user)

    authorize.assert_awaited_once_with(user, DATASET_ID)
    lookup.assert_not_called()


# --- HTTP: GET /datasets/{id}/data?content_hash=... ---------------------------


def _row(content_hash: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        name="text_" + content_hash + ".txt",
        created_at=datetime.now(timezone.utc),
        updated_at=None,
        extension="txt",
        mime_type="text/plain",
        raw_data_location=f"/tmp/{content_hash}.txt",
        dataset_id=None,
        label=None,
        external_metadata=None,
        data_size=42,
        content_hash=content_hash,
    )


MATCHING = _row(CONTENT_HASH)
PAGE = [_row(f"{i:032x}") for i in range(5)]


@pytest.fixture
def client(monkeypatch):
    module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(module, "send_telemetry", lambda *a, **k: None)

    async def _authorized(dataset_ids, permission, user):
        return [SimpleNamespace(id=DATASET_ID)]

    monkeypatch.setattr(module, "get_authorized_existing_datasets", _authorized)

    methods = importlib.import_module("cognee.modules.data.methods")
    calls = SimpleNamespace(paged=0, by_hash=[])

    async def _get_dataset_data(dataset_id, limit=None, offset=0, *, order_by="size"):
        calls.paged += 1
        return PAGE

    async def _get_dataset_data_by_content_hash(dataset_id, content_hash):
        calls.by_hash.append((dataset_id, content_hash))
        return [MATCHING] if content_hash == CONTENT_HASH else []

    monkeypatch.setattr(methods, "get_dataset_data", _get_dataset_data)
    monkeypatch.setattr(
        methods, "get_dataset_data_by_content_hash", _get_dataset_data_by_content_hash
    )

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
        c.calls = calls
        yield c


def test_content_hash_filter_returns_only_matching_items(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?content_hash={CONTENT_HASH}")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == [str(MATCHING.id)]
    assert body[0]["contentHash"] == CONTENT_HASH
    assert client.calls.by_hash == [(DATASET_ID, CONTENT_HASH)]
    assert client.calls.paged == 0, "a hash lookup must not fall back to paging"


def test_content_hash_filter_with_no_match_is_an_empty_list(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?content_hash={'0' * 32}")

    assert response.status_code == 200
    assert response.json() == []


def test_listing_without_the_filter_still_pages_and_exposes_content_hash(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(PAGE)
    assert [item["contentHash"] for item in body] == [row.content_hash for row in PAGE]
    assert client.calls.by_hash == []


def test_empty_content_hash_is_rejected(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?content_hash=")

    assert response.status_code == 422
