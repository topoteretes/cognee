"""``cognee.update()`` and ``datasets.list_data()`` must proxy after ``serve()``.

Before this, ``update()`` never consulted the remote client: it resolved the
LOCAL default user, ran a local ``datasets.delete_data`` against the remote
dataset UUID (``Dataset '<uuid>' not found``), and never touched the server.
The server already exposes ``PATCH /api/v1/update``; the SDK must call it
with the fields that route accepts — a real in-place replace, never a local
delete plus a remote add (which would mint a new id on the remote).
"""

import io
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.update.update  # noqa: F401  (bind the real submodule)
from cognee.api.v1.serve import state as serve_state
from cognee.api.v1.serve.cloud_client import CloudClient, _attach_upload
from cognee.tasks.ingestion.data_item import DataItem

update_module = sys.modules["cognee.api.v1.update.update"]


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _Session:
    def __init__(self, payload=None, status=200):
        self.calls = []
        self._payload = payload
        self._status = status

    def _record(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _Response(self._payload, self._status)

    def patch(self, url, **kwargs):
        return self._record("PATCH", url, **kwargs)

    def get(self, url, **kwargs):
        return self._record("GET", url, **kwargs)


def _fields(form):
    """(name, filename, value) triples from an aiohttp FormData."""
    return [(opts["name"], opts.get("filename"), value) for opts, _headers, value in form._fields]


# ----- CloudClient.update wire format -----


@pytest.mark.asyncio
async def test_cloud_client_update_patches_the_update_route_with_query_and_multipart():
    client = CloudClient("https://example.test", "key")
    session = _Session(payload={"status": "incremental", "regions": 1})
    client._get_session = AsyncMock(return_value=session)
    data_id, dataset_id = uuid4(), uuid4()

    result = await client.update(
        data_id=data_id,
        data="hello, updated",
        dataset_id=dataset_id,
        node_set=["repro", "second"],
        chunk_level_diff=False,
    )

    assert result == {"status": "incremental", "regions": 1}
    ((method, url, kwargs),) = session.calls
    assert method == "PATCH"
    assert url == "https://example.test/api/v1/update"
    assert kwargs["params"] == {
        "data_id": str(data_id),
        "dataset_id": str(dataset_id),
        "chunk_level_diff": "false",
    }
    fields = _fields(kwargs["data"])
    names = [name for name, _, _ in fields]
    assert names.count("data") == 1
    assert [value for name, _, value in fields if name == "node_set"] == ["repro", "second"]
    ((_, filename, body),) = [f for f in fields if f[0] == "data"]
    assert filename.startswith("text_") and filename.endswith(".txt")
    assert body.read() == b"hello, updated"


@pytest.mark.asyncio
async def test_cloud_client_update_raises_on_server_error():
    client = CloudClient("https://example.test", "key")
    client._get_session = AsyncMock(return_value=_Session(payload="nope", status=404))

    with pytest.raises(RuntimeError, match="Remote update failed \\(404\\)"):
        await client.update(data_id=uuid4(), data="x", dataset_id=uuid4())


def test_attach_upload_unwraps_data_item_and_uploads_local_file_bytes(tmp_path):
    import aiohttp

    source = tmp_path / "note.md"
    source.write_text("from disk")

    form = aiohttp.FormData()
    _attach_upload(form, DataItem(data=str(source), data_id=uuid4()))
    ((name, filename, value),) = _fields(form)
    assert name == "data"
    assert filename == "note.md"
    assert value.read() == b"from disk"
    value.close()

    form = aiohttp.FormData()
    _attach_upload(form, [f"file://{source}"])
    ((_, filename, value),) = _fields(form)
    assert filename == "note.md"
    value.close()

    form = aiohttp.FormData()
    handle = io.BytesIO(b"stream")
    handle.name = "/tmp/somewhere/stream.txt"
    _attach_upload(form, handle)
    ((_, filename, value),) = _fields(form)
    assert filename == "stream.txt"
    assert value is handle


def test_attach_upload_rejects_missing_path_and_multi_item_lists():
    import aiohttp

    with pytest.raises(FileNotFoundError):
        _attach_upload(aiohttp.FormData(), "/definitely/not/here.txt")
    with pytest.raises(ValueError, match="exactly one document"):
        _attach_upload(aiohttp.FormData(), ["a", "b"])


# ----- SDK routing -----


@pytest.mark.asyncio
async def test_update_proxies_to_remote_and_does_no_local_work():
    import cognee

    data_id, dataset_id = uuid4(), uuid4()
    client = MagicMock()
    client.update = AsyncMock(return_value={"status": "unchanged"})
    local = {
        "get_default_user": AsyncMock(),
        "incremental_update": AsyncMock(),
        "datasets": SimpleNamespace(delete_data=AsyncMock()),
        "add": AsyncMock(),
        "cognify": AsyncMock(),
    }

    with (
        patch.object(serve_state, "get_remote_client", return_value=client),
        patch.object(update_module, "get_default_user", local["get_default_user"]),
        patch.object(update_module, "incremental_update", local["incremental_update"]),
        patch.object(update_module, "datasets", local["datasets"]),
        patch.object(update_module, "add", local["add"]),
        patch.object(update_module, "cognify", local["cognify"]),
    ):
        result = await cognee.update(
            data_id=data_id,
            data="hello, updated",
            dataset_id=dataset_id,
            node_set=["repro"],
        )

    assert result == {"status": "unchanged"}
    client.update.assert_awaited_once_with(
        data_id=data_id,
        data="hello, updated",
        dataset_id=dataset_id,
        dataset_name=None,
        node_set=["repro"],
        chunk_level_diff=True,
    )
    local["get_default_user"].assert_not_awaited()
    local["incremental_update"].assert_not_awaited()
    local["datasets"].delete_data.assert_not_awaited()
    local["add"].assert_not_awaited()
    local["cognify"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_stays_local_without_a_remote_client():
    """No client → the existing local path runs, starting with user resolution."""
    data_methods_module = sys.modules["cognee.modules.data.methods"]

    with (
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch.object(update_module, "get_default_user", AsyncMock()) as get_default_user,
        patch.object(
            data_methods_module, "resolve_data_id", AsyncMock(return_value=None)
        ) as resolve,
    ):
        with pytest.raises(Exception):
            await update_module.update(data_id=uuid4(), data="x", dataset_id=uuid4())

    get_default_user.assert_awaited_once()
    resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_data_proxies_to_remote_with_attribute_access():
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id, data_id = uuid4(), uuid4()
    client = MagicMock()
    client.list_data = AsyncMock(
        return_value=[{"id": str(data_id), "name": "hello", "dataset_id": str(dataset_id)}]
    )

    with (
        patch.object(serve_state, "get_remote_client", return_value=client),
        patch.object(
            sys.modules["cognee.api.v1.datasets.datasets"], "get_default_user", AsyncMock()
        ) as gdu,
    ):
        rows = await datasets.list_data(dataset_id)

    client.list_data.assert_awaited_once_with(dataset_id)
    gdu.assert_not_awaited()
    assert [row.id for row in rows] == [str(data_id)]
    assert rows[0].name == "hello"


@pytest.mark.asyncio
async def test_cloud_client_list_data_hits_dataset_data_route():
    client = CloudClient("https://example.test/", "key")
    session = _Session(payload=[{"id": "abc"}])
    client._get_session = AsyncMock(return_value=session)
    dataset_id = uuid4()

    assert await client.list_data(dataset_id) == [{"id": "abc"}]
    ((method, url, _),) = session.calls
    assert method == "GET"
    assert url == f"https://example.test/api/v1/datasets/{dataset_id}/data"


@pytest.mark.asyncio
async def test_cloud_client_update_sends_dataset_name_instead_of_id():
    client = CloudClient("https://example.test", "key")
    session = _Session(payload={"status": "unchanged"})
    client._get_session = AsyncMock(return_value=session)
    data_id = uuid4()

    await client.update(data_id=data_id, data="x", dataset_name="repro_update")

    ((_, _, kwargs),) = session.calls
    assert kwargs["params"] == {
        "data_id": str(data_id),
        "chunk_level_diff": "true",
        "dataset_name": "repro_update",
    }


@pytest.mark.asyncio
async def test_cloud_client_update_requires_exactly_one_dataset_selector():
    client = CloudClient("https://example.test", "key")
    client._get_session = AsyncMock()

    with pytest.raises(ValueError, match="exactly one"):
        await client.update(data_id=uuid4(), data="x")
    with pytest.raises(ValueError, match="exactly one"):
        await client.update(data_id=uuid4(), data="x", dataset_id=uuid4(), dataset_name="n")


@pytest.mark.asyncio
async def test_update_forwards_dataset_name_to_remote():
    client = MagicMock()
    client.update = AsyncMock(return_value={"status": "unchanged"})
    data_id = uuid4()

    with (
        patch.object(serve_state, "get_remote_client", return_value=client),
        patch.object(update_module, "get_default_user", AsyncMock()) as gdu,
    ):
        await update_module.update(data_id=data_id, data="x", dataset_name="repro_update")

    gdu.assert_not_awaited()
    assert client.update.await_args.kwargs["dataset_name"] == "repro_update"
    assert client.update.await_args.kwargs["dataset_id"] is None


@pytest.mark.asyncio
async def test_update_rejects_neither_or_both_dataset_selectors():
    with patch.object(serve_state, "get_remote_client", return_value=MagicMock()):
        with pytest.raises(Exception, match="exactly one"):
            await update_module.update(data_id=uuid4(), data="x")
        with pytest.raises(Exception, match="exactly one"):
            await update_module.update(
                data_id=uuid4(), data="x", dataset_id=uuid4(), dataset_name="n"
            )


@pytest.mark.asyncio
async def test_local_update_resolves_dataset_name_among_writable_datasets():
    """Locally a name resolves via the write-ACL lookup and never creates a dataset."""
    data_methods_module = sys.modules["cognee.modules.data.methods"]
    dataset_id = uuid4()
    lookup = AsyncMock(return_value=[SimpleNamespace(id=dataset_id)])

    with (
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch.object(update_module, "get_default_user", AsyncMock(return_value="user")),
        patch.object(data_methods_module, "get_authorized_existing_datasets", lookup),
        patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=None)),
    ):
        with pytest.raises(Exception, match="No document found to update"):
            await update_module.update(data_id=uuid4(), data="x", dataset_name="repro")

    lookup.assert_awaited_once_with(["repro"], "write", "user")

    from cognee.modules.data.exceptions import DatasetNotFoundError

    lookup.return_value = []
    with (
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch.object(update_module, "get_default_user", AsyncMock(return_value="user")),
        patch.object(data_methods_module, "get_authorized_existing_datasets", lookup),
    ):
        with pytest.raises(DatasetNotFoundError):
            await update_module.update(data_id=uuid4(), data="x", dataset_name="missing")
