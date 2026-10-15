"""run_pipeline with selected Data rows processes only current rows of the run's dataset.

Runs on the default local stack (sqlite), no API keys.
"""

import asyncio
import importlib
import shutil
import tempfile
from pathlib import Path

import pytest

CACHED_CONFIGS = [
    ("cognee.base_config", "get_base_config"),
    ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
    ("cognee.infrastructure.databases.relational.get_relational_engine", "get_relational_engine"),
    ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
    ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
    ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
    ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
    ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
    (
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
        "create_embedding_engine",
    ),
]


def _clear_config_caches():
    for module_name, factory_name in CACHED_CONFIGS:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(scope="module")
def local_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_selected_data_test_"))
    import cognee  # (cognee's import runs load_dotenv(override=True))

    mp = pytest.MonkeyPatch()
    for key, value in {
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "CACHE_BACKEND": "sqlite",
        "MOCK_EMBEDDING": "true",
        "TELEMETRY_DISABLED": "1",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    }.items():
        mp.setenv(key, value)
    _clear_config_caches()
    yield root
    mp.undo()
    _clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


async def _add_one(text, dataset_name):
    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    await cognee.add(text, dataset_name=dataset_name, skip_connection_test=True)
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == dataset_name)
    (row,) = await get_dataset_data(dataset.id)
    return dataset, row


async def _run_selected(rows, dataset_name):
    from cognee.modules.pipelines import Task
    from cognee.modules.run_custom_pipeline import run_custom_pipeline

    seen = []

    async def record(data):
        seen.extend(data)
        return data

    await run_custom_pipeline(
        tasks=[Task(record)],
        data=rows,
        dataset=dataset_name,
        incremental_loading=True,
        skip_connection_test=True,
    )
    return seen


async def _load(data_id):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    async with get_relational_engine().get_async_session() as session:
        return await session.get(Data, data_id)


def test_row_of_another_dataset_is_rejected(local_env):
    async def scenario():
        from cognee.exceptions import CogneeValidationError

        _, foreign = await _add_one("Foreign document for the selection test.", "selected_a")
        await _add_one("Own document for the selection test.", "selected_b")

        with pytest.raises(CogneeValidationError, match="do not belong to dataset"):
            await _run_selected([foreign], "selected_b")

        assert "custom_pipeline" not in (await _load(foreign.id)).pipeline_status

    asyncio.run(scenario())


def test_row_deleted_after_selection_is_skipped(local_env):
    async def scenario():
        import cognee

        dataset, row = await _add_one("Document deleted before the run.", "selected_deleted")
        await cognee.forget(data_id=row.id, dataset_id=dataset.id)

        assert await _run_selected([row], "selected_deleted") == []

    asyncio.run(scenario())


def test_row_changed_after_selection_is_processed_as_current(local_env):
    async def scenario():
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.data.models import Data

        _, row = await _add_one("Document changed before the run.", "selected_stale")
        async with get_relational_engine().get_async_session() as session:
            (await session.get(Data, row.id)).name = "current name"
            await session.commit()

        seen = await _run_selected([row], "selected_stale")

        assert [item.name for item in seen] == ["current name"]

    asyncio.run(scenario())


def test_row_forgotten_while_the_run_waits_for_the_lock_is_skipped(local_env):
    """forget() holds the dataset lock while the run waits on it: the run reads the
    selection after forget() has deleted the row, so the row is skipped."""

    async def scenario():
        import cognee
        from cognee.infrastructure.locks import get_dataset_lock
        from cognee.modules.pipelines.operations import pipeline

        lock_module = importlib.import_module("cognee.infrastructure.locks.dataset_lock")
        dataset, row = await _add_one("Document forgotten while the run waits.", "selected_locked")

        def signalling(get_lock, event):
            async def get_lock_and_signal(dataset_id):
                lock = await get_lock(dataset_id)
                event.set()  # the caller queues on the lock right after this returns
                return lock

            return get_lock_and_signal

        forget_waiting, run_waiting = asyncio.Event(), asyncio.Event()
        mp = pytest.MonkeyPatch()
        mp.setattr(
            lock_module,
            "get_dataset_lock",
            signalling(lock_module.get_dataset_lock, forget_waiting),
        )
        mp.setattr(pipeline, "get_dataset_lock", signalling(pipeline.get_dataset_lock, run_waiting))
        try:
            # Queue forget() first and the run second, so forget() deletes the row
            # while the run is already waiting with the row selected.
            async with await get_dataset_lock(dataset.id):
                forgetting = asyncio.create_task(
                    cognee.forget(data_id=row.id, dataset_id=dataset.id)
                )
                await forget_waiting.wait()
                run = asyncio.create_task(_run_selected([row], "selected_locked"))
                await run_waiting.wait()

            await forgetting
            seen = await run
        finally:
            mp.undo()

        assert seen == []

    asyncio.run(scenario())
