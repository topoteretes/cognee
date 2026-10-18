"""Regression: a foreground ``add()`` must run the deferred DLT orphan_cleanup.

``resolve_dlt_sources`` returns a deferred ``orphan_cleanup`` that forgets rows
deleted upstream. ``add()`` used to await it only for ``run_in_background=True``,
so forget-on-source-deletion was silently broken in the default (foreground)
path for every DLT connector. This drives a document-tagged source (the path
used by Gmail and Drive) through the real add pipeline against local stores
(no LLM) and asserts a hard-deleted document is purged from cognee. Relational
manifests have stable identities and deliberately do not refresh on re-add.
"""

import logging
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user

logger = logging.getLogger(__name__)

DATASET = "widgets_ds"


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")  # no LLM/embedding ping
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    root = pathlib.Path(tmp_path)
    monkeypatch.setenv("DLT_DATA_DIR", str(root / "dlt"))  # isolate dlt pipeline state
    monkeypatch.setenv("PIPELINES_DIR", str(root / "dlt" / "pipelines"))
    monkeypatch.setenv("DB_PATH", str(root / "databases"))

    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    Container()[PipelineContext].deactivate()
    get_dlt_destination.cache_clear()
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
        logger.debug("Ignoring exception in clean_env", exc_info=True)
    finally:
        Container()[PipelineContext].deactivate()
        get_dlt_destination.cache_clear()
        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
        graph_db_config.set(None)
        vector_db_config.set(None)


def _dlt_source(rows):
    """A minimal, connector-agnostic dlt resource: merge + id PK + hard-delete."""
    import dlt

    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

    @dlt.resource(
        name="widgets",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def widgets():
        yield from rows

    setattr(widgets, DOCUMENT_SOURCE_ATTR, "test_documents")
    return widgets


async def _dlt_page_ids(user):
    """External IDs of live document-mode Data records, not staged DLT rows."""
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]
    rows = await get_dataset_data(dataset.id)
    pks = []
    for d in rows:
        ext = d.system_metadata if isinstance(d.system_metadata, dict) else {}
        if ext.get("source") == "test_documents":
            pks.append(ext.get("external_id"))
    return sorted(pks)


@pytest.mark.asyncio
async def test_foreground_add_runs_deferred_orphan_cleanup(clean_env):
    user = await get_default_user()
    kwargs = {"primary_key": "id", "write_disposition": "merge", "max_rows_per_table": 0}

    # Backfill two rows via the real (foreground) add pipeline.
    await cognee.add(
        _dlt_source(
            [
                {"id": "a", "content": "Alpha", "_deleted": False},
                {"id": "b", "content": "Beta", "_deleted": False},
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

    # An empty replacement still needs a successful completion before cleanup.
    await cognee.add(_dlt_source([{"id": "a", "_deleted": True}]), dataset_name=DATASET, **kwargs)
    assert await _dlt_page_ids(user) == []
