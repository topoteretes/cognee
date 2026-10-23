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
serve_state = importlib.import_module("cognee.api.v1.serve.state")

DATASET_ID = uuid.uuid4()
CONTENT = "the text that was added"
CONTENT_HASH = compute_content_hash(CONTENT)


# --- SDK: datasets.find_data -------------------------------------------------


@pytest.fixture
def local_mode(monkeypatch):
    """No serve() connection: find_data must query the local store."""
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)


def _user():
    return SimpleNamespace(id=uuid.uuid4(), tenant_id=None)


@pytest.mark.asyncio
async def test_find_data_hashes_content_and_looks_up_by_hash_scoped_to_the_user(local_mode):
    user = _user()
    dataset = SimpleNamespace(id=DATASET_ID)
    row = SimpleNamespace(id=uuid.uuid4(), content_hash=CONTENT_HASH)
    lookup = AsyncMock(return_value=[row])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
    ):
        found = await datasets_module.datasets.find_data(DATASET_ID, content=CONTENT, user=user)

    assert found == [row]
    # The user is handed to the lookup: matches are scoped to the caller's own
    # rows (owner + tenant), the same way ingestion dedup scopes.
    lookup.assert_awaited_once_with(DATASET_ID, CONTENT_HASH, user)


@pytest.mark.asyncio
async def test_find_data_accepts_a_precomputed_hash(local_mode):
    user = _user()
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
    lookup.assert_awaited_once_with(DATASET_ID, CONTENT_HASH, user)


@pytest.mark.asyncio
async def test_find_data_normalizes_a_precomputed_hash(local_mode):
    """An uppercase or padded digest (e.g. from .NET) must match the stored lowercase."""
    user = _user()
    dataset = SimpleNamespace(id=DATASET_ID)
    lookup = AsyncMock(return_value=[])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
    ):
        await datasets_module.datasets.find_data(
            DATASET_ID, content_hash=f"  {CONTENT_HASH.upper()}\n", user=user
        )

    lookup.assert_awaited_once_with(DATASET_ID, CONTENT_HASH, user)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_hash", ["", "abc", CONTENT_HASH[:-1], "g" * 32, CONTENT_HASH + "0"])
async def test_find_data_rejects_a_malformed_hash(local_mode, bad_hash):
    """A value that is not an MD5 digest is an error, not a confident 'not found'."""
    with pytest.raises(ValueError, match="32-character hex"):
        await datasets_module.datasets.find_data(DATASET_ID, content_hash=bad_hash, user=_user())


@pytest.mark.asyncio
async def test_find_data_requires_exactly_one_selector(local_mode):
    user = _user()
    with pytest.raises(ValueError):
        await datasets_module.datasets.find_data(DATASET_ID, user=user)
    with pytest.raises(ValueError):
        await datasets_module.datasets.find_data(
            DATASET_ID, content=CONTENT, content_hash=CONTENT_HASH, user=user
        )


@pytest.mark.asyncio
async def test_find_data_checks_read_access_before_looking_up(local_mode):
    user = _user()
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


@pytest.mark.asyncio
async def test_find_data_refuses_strings_add_would_resolve_as_references(local_mode, tmp_path):
    """``content`` is the literal text; a path/URL hashed as text would match nothing.

    ``add()`` ingests the file or page behind such a string and stores the
    fetched bytes' hash, so hashing the string itself would silently answer
    "not present". Refuse it and point at ``content_hash`` instead.
    """
    existing_file = tmp_path / "report.txt"
    existing_file.write_text("report body")

    lookup = AsyncMock()
    with patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup):
        for reference in (
            str(existing_file),
            existing_file.as_uri(),
            "s3://bucket/path/report.pdf",
            "https://example.com/page",
            "http://example.com/page",
        ):
            # The message names the escape hatch for a string stored as text.
            with pytest.raises(ValueError, match=r"literal text.*compute_content_hash"):
                await datasets_module.datasets.find_data(
                    DATASET_ID, content=reference, user=_user()
                )

    lookup.assert_not_called()


@pytest.mark.asyncio
async def test_find_data_hashes_a_slash_prefixed_note_that_is_not_a_file_as_text(local_mode):
    """Same rule as add(): a "/"-prefixed string that names no file is text."""
    note = "/remember to call Bob"
    user = _user()
    dataset = SimpleNamespace(id=DATASET_ID)
    lookup = AsyncMock(return_value=[])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
    ):
        await datasets_module.datasets.find_data(DATASET_ID, content=note, user=user)

    lookup.assert_awaited_once_with(DATASET_ID, compute_content_hash(note), user)


def _remote_row(content_hash: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": "text_" + content_hash + ".txt",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": None,
        "extension": "txt",
        "mimeType": "text/plain",
        "rawDataLocation": f"/srv/{content_hash}.txt",
        "datasetId": str(DATASET_ID),
        "label": None,
        "externalMetadata": None,
        "dataSize": 11,
        "contentHash": content_hash,
    }


@pytest.mark.asyncio
async def test_find_data_routes_to_the_remote_instance_under_serve(monkeypatch):
    """Like list_data: connected via serve(), the dataset lives on the server."""
    remote_row = _remote_row(CONTENT_HASH)
    client = SimpleNamespace(list_data=AsyncMock(return_value=[remote_row]))
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    authorize = AsyncMock()
    lookup = AsyncMock()
    with (
        patch.object(datasets_module, "get_authorized_dataset", authorize),
        patch.object(datasets_module, "get_dataset_data_by_content_hash", lookup),
    ):
        found = await datasets_module.datasets.find_data(DATASET_ID, content=CONTENT)

    client.list_data.assert_awaited_once_with(DATASET_ID, content_hash=CONTENT_HASH)
    # Parsed through the same DTO the server serializes with, as list_data does.
    assert [str(row.id) for row in found] == [remote_row["id"]]
    assert found[0].content_hash == CONTENT_HASH
    assert found[0].mime_type == "text/plain"
    # The local store is never consulted.
    authorize.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.asyncio
async def test_find_data_under_serve_ignores_files_on_this_machine(monkeypatch, tmp_path):
    """Under serve(), add() resolved the string on the server, not here.

    A note that happens to name a file on the client was stored as text on the
    server, so it is hashed as text and sent, not refused.
    """
    local_file = tmp_path / "README.md"
    local_file.write_text("a file on the client")
    note = str(local_file)
    client = SimpleNamespace(list_data=AsyncMock(return_value=[]))
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    assert await datasets_module.datasets.find_data(DATASET_ID, content=note) == []

    client.list_data.assert_awaited_once_with(DATASET_ID, content_hash=compute_content_hash(note))


@pytest.mark.asyncio
async def test_find_data_under_serve_still_refuses_urls(monkeypatch):
    """URL schemes are references wherever add() ran, so they stay refused remotely."""
    client = SimpleNamespace(list_data=AsyncMock())
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    for reference in ("https://example.com/page", "s3://bucket/report.pdf"):
        with pytest.raises(ValueError, match="compute_content_hash"):
            await datasets_module.datasets.find_data(DATASET_ID, content=reference)

    client.list_data.assert_not_called()


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

    async def _get_dataset_data_by_content_hash(
        dataset_id, content_hash, user, *, limit=None, offset=0
    ):
        calls.by_hash.append((dataset_id, content_hash, user.id, limit, offset))
        return [MATCHING] if content_hash == CONTENT_HASH else []

    monkeypatch.setattr(methods, "get_dataset_data", _get_dataset_data)
    monkeypatch.setattr(
        methods, "get_dataset_data_by_content_hash", _get_dataset_data_by_content_hash
    )

    app = FastAPI()
    app.include_router(module.get_datasets_router(), prefix="/api/v1/datasets")

    user_id = str(uuid.uuid4())

    async def _user():
        return SimpleNamespace(
            id=user_id,
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    app.dependency_overrides[get_authenticated_user] = _user
    with TestClient(app) as c:
        c.calls = calls
        c.user_id = user_id
        yield c


def test_content_hash_filter_returns_only_matching_items(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?content_hash={CONTENT_HASH}")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == [str(MATCHING.id)]
    assert body[0]["contentHash"] == CONTENT_HASH
    # Scoped to the authenticated user, paged with the route's defaults.
    assert client.calls.by_hash == [(DATASET_ID, CONTENT_HASH, client.user_id, 100, 0)]
    assert client.calls.paged == 0, "a hash lookup must not fall back to the plain listing"


def test_content_hash_filter_normalizes_case(client):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?content_hash={CONTENT_HASH.upper()}")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [str(MATCHING.id)]
    assert client.calls.by_hash[0][1] == CONTENT_HASH


def test_content_hash_filter_honours_limit_and_offset(client):
    response = client.get(
        f"/api/v1/datasets/{DATASET_ID}/data?content_hash={CONTENT_HASH}&limit=7&offset=14"
    )

    assert response.status_code == 200
    assert client.calls.by_hash == [(DATASET_ID, CONTENT_HASH, client.user_id, 7, 14)]


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


@pytest.mark.parametrize("bad_hash", ["", "abc", CONTENT_HASH[:-1], "g" * 32, CONTENT_HASH + "0"])
def test_malformed_content_hash_is_rejected(client, bad_hash):
    response = client.get(f"/api/v1/datasets/{DATASET_ID}/data?content_hash={bad_hash}")

    assert response.status_code == 422
    assert client.calls.by_hash == []
