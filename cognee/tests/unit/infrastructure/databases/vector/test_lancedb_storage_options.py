"""LanceDB gets the Azure storage options for an az:// store: in-process, through the
subprocess worker, and in the connect replayed after a worker respawn. Other stores get
none and keep reading their credentials from the environment."""

import importlib
from types import SimpleNamespace

import pytest

adapter_module = importlib.import_module(
    "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
)

AZURE_OPTIONS = {"azure_storage_account_name": "acct", "azure_storage_account_key": "a2V5"}


@pytest.fixture
def azure_settings(monkeypatch):
    from cognee.infrastructure.files.storage import azure_config

    settings = SimpleNamespace(lance_storage_options=lambda: AZURE_OPTIONS)
    monkeypatch.setattr(azure_config, "get_azure_config", lambda: settings)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("az://box/system/databases/lancedb", AZURE_OPTIONS),
        ("s3://bucket/system/databases/lancedb", None),
        ("/home/me/.cognee_system/databases/lancedb", None),
        (None, None),
    ],
)
def test_only_azure_stores_get_storage_options(azure_settings, url, expected):
    assert adapter_module._storage_options_for(url) == expected


@pytest.mark.asyncio
async def test_in_process_connect_passes_the_options(azure_settings, monkeypatch):
    calls = []

    async def fake_connect_async(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(adapter_module.lancedb, "connect_async", fake_connect_async)
    adapter = adapter_module.LanceDBAdapter("az://box/v", None, embedding_engine=None)

    await adapter.get_connection()

    assert calls == [("az://box/v", {"api_key": None, "storage_options": AZURE_OPTIONS})]


class _RecordingSession:
    """Stands in for the worker session: records requests and replay steps."""

    def __init__(self):
        self.requests, self.replay_steps = [], []

    async def call_async(self, request):
        self.requests.append(request)

    def add_replay_step(self, step):
        self.replay_steps.append(step)


@pytest.mark.asyncio
async def test_worker_connect_and_its_replay_carry_the_options():
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        RemoteLanceDBConnection,
    )

    session = _RecordingSession()
    connection = RemoteLanceDBConnection(session, "az://box/v", None, AZURE_OPTIONS)

    await connection.connect()

    replayed = session.replay_steps[0].make_request()
    for request in (session.requests[0], replayed):
        assert request.kwargs["storage_options"] == AZURE_OPTIONS


@pytest.mark.asyncio
async def test_worker_opens_the_store_with_the_options(monkeypatch):
    import lancedb

    from cognee_db_workers import lancedb_worker
    from cognee_db_workers.harness import Request
    from cognee_db_workers.lancedb_protocol import OP_CONNECT

    calls = []

    async def fake_connect_async(url, **kwargs):
        calls.append(kwargs["storage_options"])
        return SimpleNamespace()

    monkeypatch.setattr(lancedb, "connect_async", fake_connect_async)
    registry = SimpleNamespace(register_at=lambda handle, value: None)
    kwargs = {"url": "az://box/v", "api_key": None, "storage_options": AZURE_OPTIONS}

    await lancedb_worker._op_connect(registry, Request(op=OP_CONNECT, kwargs=kwargs))

    assert calls == [AZURE_OPTIONS]
