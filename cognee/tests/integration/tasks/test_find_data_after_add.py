"""``datasets.find_data`` finds exactly what ``add()`` stored, through real ingestion.

The lookup's promise is that ``compute_content_hash(content)`` equals the
``content_hash`` ``add()`` records. The unit tests pin the formula against
``TextData``; this runs the real path (``add`` -> ``ingest_data`` ->
``save_data_to_file``) so a normalization added anywhere on it would fail here
instead of making every lookup quietly answer "not present". No cognify, no LLM.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def find_data_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_find_data_test_"))

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

    yield root

    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


async def _dataset(name, user):
    from cognee.modules.data.methods import get_datasets

    return next(d for d in await get_datasets(user.id) if d.name == name)


async def _scenario(root: Path):
    import cognee
    from cognee.modules.data.content_hash import compute_content_hash
    from cognee.modules.users.methods import get_default_user
    from cognee.tasks.ingestion.data_item import DataItem

    text = "The quarterly review moved to Thursday afternoon."
    report = root / "report.txt"
    report.write_bytes(b"Revenue grew twelve percent in the third quarter.\n")
    # A URL-shaped string cognee keeps verbatim (as DLT rows and session
    # bridging do): find_data(content=...) refuses it, content_hash= finds it.
    literal_url = "https://example.com/status"

    await cognee.add(text, dataset_name="alpha")
    await cognee.add(str(report), dataset_name="alpha")
    await cognee.add(DataItem(data=literal_url, literal_text=True), dataset_name="alpha")
    await cognee.add(text, dataset_name="beta")

    user = await get_default_user()
    alpha = await _dataset("alpha", user)
    beta = await _dataset("beta", user)

    # add(text) -> find_data(content=text): the row add() created.
    found = await cognee.datasets.find_data(alpha.id, content=text, user=user)
    assert len(found) == 1
    assert found[0].content_hash == compute_content_hash(text)
    assert str(found[0].dataset_id) == str(alpha.id)

    # add(path) stores the file's bytes; find_data(content=bytes) finds that row.
    found_file = await cognee.datasets.find_data(alpha.id, content=report.read_bytes(), user=user)
    assert len(found_file) == 1
    assert found_file[0].id != found[0].id

    # The same text in another dataset is another row, and each lookup stays
    # in its own dataset.
    found_beta = await cognee.datasets.find_data(beta.id, content=text, user=user)
    assert len(found_beta) == 1
    assert found_beta[0].id != found[0].id
    assert str(found_beta[0].dataset_id) == str(beta.id)

    # A literal URL-shaped string: refused as content=, found through the hash.
    with pytest.raises(ValueError, match="compute_content_hash"):
        await cognee.datasets.find_data(alpha.id, content=literal_url, user=user)
    found_literal = await cognee.datasets.find_data(
        alpha.id, content_hash=compute_content_hash(literal_url), user=user
    )
    assert len(found_literal) == 1

    assert await cognee.datasets.find_data(alpha.id, content="never added", user=user) == []


def test_find_data_finds_what_add_stored(find_data_env):
    asyncio.run(_scenario(find_data_env))
