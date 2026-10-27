"""remember(session_id=...) writes a session's Q&A vectors into the dataset's store.

With backend access control on (the default), a search turn answers inside its
dataset's database context, so the session's Q&A vectors land in that dataset's
isolated vector store, and session history reads them from there. The two
remember write paths (plain text and typed entries) must write to the same
store, or the turns they record are invisible to session history's vector lane.

Real stores throughout: SQLite relational, the sqlite session cache, one LanceDB
per dataset, mock embeddings (zero vectors, no network).
"""

import asyncio
import contextvars
import pathlib
from collections.abc import Awaitable, Callable

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.session.session_embeddings import SESSION_QA_VECTOR_COLLECTION
from cognee.memory import QAEntry
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user


@pytest_asyncio.fixture
async def isolated_dataset(request, tmp_path, monkeypatch):
    """A fresh cognee system with access control on, one user and one dataset."""
    pytest.importorskip("ladybug")

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("IMPROVE_AUTO_ENABLED", "false")
    root = pathlib.Path(tmp_path) / request.node.name

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)
    cognee.config.set_graph_db_config(
        {"graph_database_provider": "ladybug", "graph_dataset_database_handler": "ladybug"}
    )
    cognee.config.set_vector_db_config(
        {"vector_db_provider": "lancedb", "vector_dataset_database_handler": "lancedb"}
    )
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.set_migration_db_config({"migration_db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await engine_setup()
    user = await get_default_user()
    dataset = await create_authorized_dataset("sessions", user)
    yield user, dataset


async def _as_fresh_request(call: Callable[[], Awaitable]):
    """Run ``call`` the way the API or the CLI runs it: a new task with no dataset
    context inherited from this test. Inside one task the per-dataset config set
    by ``set_database_global_context_variables`` outlives the block (released at
    task end), which would mask where a write really went."""
    return await asyncio.create_task(call(), context=contextvars.Context())


async def _session_vector_rows() -> int:
    """Rows in the session Q&A collection of the vector store the current context selects."""
    vector_engine = await get_vector_engine_async()
    if not await vector_engine.has_collection(SESSION_QA_VECTOR_COLLECTION):
        return 0
    connection = await vector_engine.get_connection()
    collection = await connection.open_table(SESSION_QA_VECTOR_COLLECTION)
    return await collection.count_rows()


async def _assert_only_in_dataset_store(user, dataset, rows: int) -> None:
    assert await _as_fresh_request(_session_vector_rows) == 0, "nothing in the global store"
    async with set_database_global_context_variables(dataset.id, user.id):
        assert await _session_vector_rows() == rows


@pytest.mark.asyncio
async def test_plain_text_remember_writes_into_the_dataset_store(isolated_dataset):
    user, dataset = isolated_dataset

    async def remember_turn():
        await cognee.remember(
            "The door code is 8919-6729.", session_id="s1", dataset_name=dataset.name
        )

    await _as_fresh_request(remember_turn)

    await _assert_only_in_dataset_store(user, dataset, rows=1)


@pytest.mark.asyncio
async def test_typed_entry_remember_writes_into_the_dataset_store(isolated_dataset):
    user, dataset = isolated_dataset

    async def remember_entry():
        await cognee.remember(
            QAEntry(question="What is the door code?", answer="8919-6729"),
            session_id="s2",
            dataset_name=dataset.name,
        )

    await _as_fresh_request(remember_entry)

    await _assert_only_in_dataset_store(user, dataset, rows=1)


@pytest.mark.asyncio
async def test_a_second_turn_joins_the_first_in_the_dataset_store(isolated_dataset):
    """Plain and typed writes to one session end up side by side, where history reads."""
    user, dataset = isolated_dataset

    async def two_turns():
        await cognee.remember("first turn", session_id="s3", dataset_name=dataset.name)
        await cognee.remember(
            QAEntry(question="q", answer="second turn"), session_id="s3", dataset_name=dataset.name
        )

    await _as_fresh_request(two_turns)

    await _assert_only_in_dataset_store(user, dataset, rows=2)
