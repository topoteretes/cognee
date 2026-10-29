"""delete_session() cleans the Q&A vectors where the session's turns wrote them.

With backend access control on (the default), a search turn answers inside its
dataset's database context, so the session's Q&A vectors land in that dataset's
isolated vector store, not the global one. The delete must run its vector
cleanup in the same context or it strips nothing, raises nothing, and reports
success while the session keeps answering recall.

Real stores throughout: SQLite relational, the sqlite session cache, one LanceDB
per dataset, mock embeddings (zero vectors, no network).
"""

import asyncio
import contextvars
import importlib
import pathlib
from collections.abc import Awaitable, Callable
from uuid import UUID

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.locks import acquire_improve_lock_many, release_improve_lock_many
from cognee.infrastructure.locks.session_lock import improve_lock_keys
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_embeddings import SESSION_QA_VECTOR_COLLECTION
from cognee.memory.entries import QAEntry
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.session_lifecycle.delete_session import delete_session
from cognee.modules.users.methods import create_user, get_default_user

delete_module = importlib.import_module("cognee.modules.session_lifecycle.delete_session")


@pytest_asyncio.fixture
async def isolated_dataset(request, tmp_path, monkeypatch):
    """A fresh cognee system with access control on, one user and one dataset."""
    pytest.importorskip("ladybug")

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
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
    """Run ``call`` the way the API route or the CLI runs: a new task with no
    dataset context inherited from this test. Inside one task the per-dataset
    config set by ``set_database_global_context_variables`` outlives the block
    (it is released at task end), which would mask the bug under test."""
    return await asyncio.create_task(call(), context=contextvars.Context())


async def _session_vector_rows() -> int:
    """Rows in the session Q&A collection of the vector store the current context selects."""
    vector_engine = await get_vector_engine_async()
    if not await vector_engine.has_collection(SESSION_QA_VECTOR_COLLECTION):
        return 0
    connection = await vector_engine.get_connection()
    collection = await connection.open_table(SESSION_QA_VECTOR_COLLECTION)
    return await collection.count_rows()


async def _answer_turn(user_id: UUID, dataset_id: UUID, session_id: str) -> None:
    """What a search turn does: index the Q&A inside the dataset's database context."""
    async with set_database_global_context_variables(dataset_id, user_id):
        session_manager = get_session_manager(dataset_id=dataset_id)
        await session_manager.add_qa(
            user_id=str(user_id),
            question="What is the door code?",
            context="",
            answer="8919-6729",
            session_id=session_id,
        )


@pytest.mark.asyncio
async def test_delete_session_strips_vectors_from_the_dataset_store(isolated_dataset):
    user, dataset = isolated_dataset
    await _answer_turn(user.id, dataset.id, "s1")

    # The vectors are in the dataset's own store, and only there.
    async with set_database_global_context_variables(dataset.id, user.id):
        assert await _session_vector_rows() == 1
    assert await _as_fresh_request(_session_vector_rows) == 0, (
        "the global store must hold no session vectors"
    )

    # The API route and the CLI call this with no dataset context at all.
    assert await _as_fresh_request(lambda: delete_session(user.id, "s1")) is True

    async with set_database_global_context_variables(dataset.id, user.id):
        assert await _session_vector_rows() == 0


@pytest.mark.asyncio
async def test_without_the_dataset_context_the_vectors_survive(isolated_dataset, monkeypatch):
    """The regression this guards: cleaning in the ambient context reaches the wrong store."""
    user, dataset = isolated_dataset
    await _answer_turn(user.id, dataset.id, "s2")

    async def no_datasets(user_id, session_id):
        return []

    monkeypatch.setattr(delete_module, "_session_vector_datasets", no_datasets)
    # Cache content went, so it reports True; the vectors in the dataset store did not.
    assert await _as_fresh_request(lambda: delete_session(user.id, "s2")) is True

    async with set_database_global_context_variables(dataset.id, user.id):
        assert await _session_vector_rows() == 1


@pytest.mark.asyncio
async def test_delete_session_strips_vectors_a_typed_entry_left_in_the_global_store(
    isolated_dataset,
):
    """A typed remember() entry binds the session to its dataset but indexes the
    Q&A in the global store, so the delete must clean that store too."""
    user, dataset = isolated_dataset
    entry = QAEntry(question="What is the door code?", answer="8919-6729")
    await _as_fresh_request(
        lambda: cognee.remember(
            entry, session_id="s3", dataset_id=dataset.id, user=user, self_improvement=False
        )
    )
    assert await _as_fresh_request(_session_vector_rows) == 1

    assert await _as_fresh_request(lambda: delete_session(user.id, "s3")) is True

    assert await _as_fresh_request(_session_vector_rows) == 0
    async with set_database_global_context_variables(dataset.id, user.id):
        assert await _session_vector_rows() == 0


async def _dataset_vector_rows(user_id: UUID, dataset_id: UUID) -> int:
    async def count():
        async with set_database_global_context_variables(dataset_id, user_id):
            return await _session_vector_rows()

    return await _as_fresh_request(count)


@pytest.mark.asyncio
async def test_delete_session_strips_vectors_from_every_dataset_it_answered_in(
    isolated_dataset,
):
    """A recall over several datasets answers, and indexes, once per dataset;
    the session record names only the first."""
    user, dataset = isolated_dataset
    other = await create_authorized_dataset("other", user)
    await _answer_turn(user.id, dataset.id, "s4")
    await _answer_turn(user.id, other.id, "s4")
    assert await _dataset_vector_rows(user.id, other.id) == 1

    assert await _as_fresh_request(lambda: delete_session(user.id, "s4")) is True

    assert await _dataset_vector_rows(user.id, dataset.id) == 0
    assert await _dataset_vector_rows(user.id, other.id) == 0


@pytest.mark.asyncio
async def test_a_session_id_naming_another_users_dataset_does_not_reach_it(isolated_dataset):
    user, dataset = isolated_dataset
    stranger = await create_user("stranger@example.com", "password")
    foreign = await create_authorized_dataset("foreign", stranger)

    targets = await delete_module._session_vector_datasets(user.id, f"x_{foreign.id}")

    assert foreign.id not in targets
    assert dataset.id in targets


@pytest.mark.asyncio
async def test_delete_session_waits_for_an_improve_holding_the_session(isolated_dataset):
    """An improve bridging the session would write its watermarks back after
    the delete; the delete waits for its claim instead."""
    user, dataset = isolated_dataset
    await _answer_turn(user.id, dataset.id, "s5")
    keys = improve_lock_keys(["s5"], user.id)
    await acquire_improve_lock_many(keys)
    try:
        delete = asyncio.create_task(delete_session(user.id, "s5"))
        await asyncio.sleep(0.5)
        assert not delete.done()
    finally:
        await release_improve_lock_many(keys)
    assert await asyncio.wait_for(delete, 30) is True
