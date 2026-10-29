"""Re-adding text under another name leaves no stored object behind.

Named text is stored under ``<content hash>/<name>.txt``, so the same content
under a new name is written to a new key. Whether ingestion then skips the item
(its content is already processed) or re-ingests it (``incremental_loading`` and
``data_cache`` off), the object no Data row points at must be released, or it
survives ``forget()`` and the text outlives its deletion. Real ingestion and storage, no cognify, no LLM.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def storage_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_named_text_cleanup_"))

    import cognee  # (cognee's import loads .env; the env below overrides it)

    def clear_config_caches():
        import importlib

        for module_name, factory_name in [
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            (
                "cognee.infrastructure.databases.relational.get_relational_engine",
                "get_relational_engine",
            ),
            ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
            ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
            ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ]:
            try:
                getattr(importlib.import_module(module_name), factory_name).cache_clear()
            except (ImportError, AttributeError):
                pass

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
    clear_config_caches()

    yield root / "data"

    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


def _stored_files(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(
        os.path.relpath(os.path.join(directory, name), data_root)
        for directory, _, names in os.walk(data_root)
        for name in names
    )


async def _add_named(name: str, dataset_name: str, incremental_loading: bool = True):
    import cognee
    from cognee.tasks.ingestion.data_item import DataItem

    await cognee.add(
        DataItem(data="the same body", name=name, literal_text=True),
        dataset_name=dataset_name,
        # Both off takes the regular path, which re-ingests instead of skipping.
        incremental_loading=incremental_loading,
        data_cache=incremental_loading,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental_loading", [True, False], ids=["skipped", "re_ingested"])
async def test_the_same_text_under_a_new_name_leaves_nothing_after_forget(
    storage_env, incremental_loading
):
    import cognee

    await _add_named("Old title", "named")
    await _add_named("New title", "named", incremental_loading=incremental_loading)

    names = [path.rsplit("/", 1)[-1] for path in _stored_files(storage_env)]
    # One original object for the one document, whichever name it kept.
    assert len([name for name in names if name in ("Old title.txt", "New title.txt")]) == 1

    await cognee.forget(dataset="named")

    assert _stored_files(storage_env) == []


@pytest.mark.asyncio
async def test_re_adding_under_the_same_name_keeps_the_stored_object(storage_env):
    await _add_named("Title", "named")
    before = _stored_files(storage_env)

    await _add_named("Title", "named")

    assert _stored_files(storage_env) == before
    assert any(path.endswith("/Title.txt") for path in before)
