"""Regression: a foreground ``add()`` must run the deferred DLT orphan_cleanup.

``resolve_dlt_sources`` returns a deferred ``orphan_cleanup`` that forgets rows
deleted upstream. ``add()`` used to await it only for ``run_in_background=True``,
so forget-on-source-deletion was silently broken in the default (foreground)
path for every DLT connector. This drives the real add pipeline twice against
local stores (no LLM) and asserts a hard-deleted row is purged from cognee.
"""

import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user

DATASET = "widgets_ds"


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")  # no LLM/embedding ping
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    root = pathlib.Path(tmp_path)
    monkeypatch.setenv("DLT_DATA_DIR", str(root / "dlt"))  # isolate dlt pipeline state

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

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _dlt_source(rows):
    """A minimal, connector-agnostic dlt resource: merge + id PK + hard-delete."""
    import dlt

    @dlt.resource(
        name="widgets",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def widgets():
        yield from rows

    return widgets


async def _dlt_page_ids(user):
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]
    rows = await get_dataset_data(dataset.id)
    return sorted(
        d.external_metadata.get("primary_key_value")
        for d in rows
        if isinstance(d.external_metadata, dict) and d.external_metadata.get("source") == "dlt"
    )


@pytest.mark.asyncio
async def test_foreground_add_runs_deferred_orphan_cleanup(clean_env):
    user = await get_default_user()
    kwargs = dict(primary_key="id", write_disposition="merge", max_rows_per_table=0)

    # Backfill two rows via the real (foreground) add pipeline.
    await cognee.add(
        _dlt_source(
            [
                {"id": "a", "body": "Alpha", "_deleted": False},
                {"id": "b", "body": "Beta", "_deleted": False},
            ]
        ),
        dataset_name=DATASET,
        **kwargs,
    )
    assert await _dlt_page_ids(user) == ["a", "b"]

    # Foreground re-sync: 'b' is hard-deleted upstream. Before the fix, the
    # foreground path never awaited orphan_cleanup, so 'b' lingered in cognee.
    await cognee.add(_dlt_source([{"id": "b", "_deleted": True}]), dataset_name=DATASET, **kwargs)
    assert await _dlt_page_ids(user) == ["a"]  # 'b' forgotten by foreground orphan_cleanup
