"""remember(session_id=...) writes a session's Q&A vectors into the dataset's store.

With backend access control on (the default), a search turn answers inside its
dataset's database context, so the session's Q&A vectors land in that dataset's
isolated vector store, and session history reads them from there. The two
remember write paths (plain text and typed entries) must write to the same
store, or the turns they record are invisible to session history's vector lane.
A plain-text write follows the session's binding exactly as a typed entry does.

Real stores throughout: SQLite relational, the sqlite session cache, one LanceDB
per dataset, mock embeddings (zero vectors, no network).
"""

import asyncio
import contextvars
import pathlib
from collections.abc import Awaitable, Callable
from uuid import uuid4

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
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.data.methods import create_authorized_dataset, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.migrations import startup as migrations_startup
from cognee.modules.migrations.startup import MigrationError
from cognee.modules.session_lifecycle.metrics import get_session_row
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import create_user, get_default_user


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


async def _dataset_store_rows(user, dataset_id) -> int:
    async def count():
        async with set_database_global_context_variables(dataset_id, user.id):
            return await _session_vector_rows()

    return await _as_fresh_request(count)


@pytest.mark.asyncio
async def test_plain_text_remember_binds_an_unbound_session(isolated_dataset):
    """The first plain-text write binds the session, like a typed entry's does."""
    user, dataset = isolated_dataset

    async def remember_turn():
        await cognee.remember("first turn", session_id="s4", dataset_name=dataset.name)

    await _as_fresh_request(remember_turn)

    row = await get_session_row(session_id="s4", user_id=user.id)
    assert row is not None and row.dataset_id == dataset.id


@pytest.mark.asyncio
async def test_plain_text_on_a_bound_session_follows_its_binding(isolated_dataset):
    """Without a dataset name, a plain-text turn goes to the session's dataset, not the default."""
    user, dataset = isolated_dataset

    async def two_turns():
        await cognee.remember(
            QAEntry(question="q", answer="bound turn"), session_id="s5", dataset_name=dataset.name
        )
        result = await cognee.remember("later turn", session_id="s5")
        assert result.dataset_id == str(dataset.id)
        assert result.dataset_name == dataset.name

    await _as_fresh_request(two_turns)

    await _assert_only_in_dataset_store(user, dataset, rows=2)
    assert await get_datasets_by_name([DEFAULT_DATASET_NAME], user.id) == []


@pytest.mark.asyncio
async def test_plain_text_naming_another_dataset_on_a_bound_session_is_refused(isolated_dataset):
    user, dataset = isolated_dataset
    other = await create_authorized_dataset("other", user)

    async def remember_turn():
        await cognee.remember("first turn", session_id="s6", dataset_name=dataset.name)

    async def remember_elsewhere():
        await cognee.remember("misrouted turn", session_id="s6", dataset_name=other.name)

    await _as_fresh_request(remember_turn)
    with pytest.raises(ValueError, match="is bound to dataset"):
        await _as_fresh_request(remember_elsewhere)

    await _assert_only_in_dataset_store(user, dataset, rows=1)
    assert await _dataset_store_rows(user, other.id) == 0


@pytest.mark.asyncio
async def test_plain_text_needs_write_access_to_an_explicit_dataset_id(isolated_dataset):
    """An explicit dataset_id is authorized before the session write enters its store."""
    _, dataset = isolated_dataset
    outsider = await create_user(f"outsider-{uuid4()}@example.com", "outsider-password")

    async def remember_turn():
        await cognee.remember(
            "intruding turn", session_id="s7", dataset_id=dataset.id, user=outsider
        )

    with pytest.raises(PermissionDeniedError):
        await _as_fresh_request(remember_turn)


@pytest.mark.asyncio
async def test_plain_text_accepts_a_returned_dataset_id_back(isolated_dataset):
    """RememberResult.dataset_id is a str; passing it back names the bound dataset."""
    user, dataset = isolated_dataset

    async def two_turns():
        first = await cognee.remember("first turn", session_id="s8", dataset_name=dataset.name)
        await cognee.remember("second turn", session_id="s8", dataset_id=first.dataset_id)

    await _as_fresh_request(two_turns)

    await _assert_only_in_dataset_store(user, dataset, rows=2)


def _fail_migrations_for(monkeypatch, dataset_id) -> None:
    """Report ``dataset_id``'s databases as failed by the startup migrations."""

    async def failed_run():
        return [str(dataset_id)]

    monkeypatch.setattr(migrations_startup, "run_migrations", failed_run)


@pytest.mark.asyncio
async def test_migration_gate_blocks_on_the_bound_dataset(isolated_dataset, monkeypatch):
    """A failed migration of the session's dataset blocks the write, even when the
    call names no dataset (which would otherwise mean main_dataset)."""
    user, dataset = isolated_dataset

    async def bind():
        await cognee.remember("first turn", session_id="s9", dataset_name=dataset.name)

    async def write_unnamed():
        await cognee.remember("blocked turn", session_id="s9")

    await _as_fresh_request(bind)
    _fail_migrations_for(monkeypatch, dataset.id)
    with pytest.raises(MigrationError):
        await _as_fresh_request(write_unnamed)

    await _assert_only_in_dataset_store(user, dataset, rows=1)


@pytest.mark.asyncio
async def test_migration_gate_ignores_an_unrelated_dataset(isolated_dataset, monkeypatch):
    """A failed migration of main_dataset does not block a session bound elsewhere."""
    user, dataset = isolated_dataset
    main = await create_authorized_dataset(DEFAULT_DATASET_NAME, user)

    async def bind():
        await cognee.remember("first turn", session_id="s10", dataset_name=dataset.name)

    async def write_unnamed():
        await cognee.remember("healthy turn", session_id="s10")

    await _as_fresh_request(bind)
    _fail_migrations_for(monkeypatch, main.id)
    await _as_fresh_request(write_unnamed)

    await _assert_only_in_dataset_store(user, dataset, rows=2)
