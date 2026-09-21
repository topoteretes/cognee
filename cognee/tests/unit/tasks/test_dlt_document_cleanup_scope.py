"""Document orphan cleanup requires evidence for the exact staging table."""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.tasks.ingestion.dlt_row_data import DltRows
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")


@pytest.mark.asyncio
@pytest.mark.parametrize("tables,cleans", [([], False), (["drive_a"], True)])
async def test_empty_read_cleans_only_a_successfully_loaded_table(tables, cleans):
    dlt = pytest.importorskip("dlt")

    @dlt.resource(name="drive_a")
    def source():
        yield from ()

    resource = source()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "google_drive")
    with (
        patch.object(
            resolve,
            "ingest_dlt_source",
            new=AsyncMock(return_value=DltRows([], loaded_tables=tables)),
        ),
        patch.object(resolve, "_delete_dlt_orphans", new=AsyncMock()) as delete,
    ):
        items, cleanup = await resolve.resolve_dlt_sources(
            resource,
            "dataset",
            SimpleNamespace(id=uuid4()),
            dataset_id=uuid4(),
        )
        assert items == []
        assert (cleanup is not None) is cleans
        if cleanup is not None:
            await cleanup()
            assert delete.await_args.kwargs["document_scopes"] == {("google_drive", "drive_a")}
            assert delete.await_args.args[2] == set()
        else:
            delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_table_cleanup_retains_other_tables_and_unscoped_legacy_rows():
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    owned = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "google_drive", "table_name": "drive_a"}
    )
    other = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "google_drive", "table_name": "drive_b"}
    )
    legacy = SimpleNamespace(id=uuid4(), system_metadata={"source": "google_drive"})

    @asynccontextmanager
    async def context(*args):
        yield

    with (
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch(
            "cognee.modules.data.methods.get_dataset_data.get_dataset_data",
            new=AsyncMock(return_value=[owned, other, legacy]),
        ),
        patch("cognee.context_global_variables.set_database_global_context_variables", context),
        patch(
            "cognee.modules.graph.methods.delete_data_nodes_and_edges.delete_data_nodes_and_edges",
            new=AsyncMock(return_value=SimpleNamespace(node_ids=[], edge_ids=[])),
        ),
        patch("cognee.modules.data.methods.delete_data.delete_data", new=AsyncMock()) as delete,
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.invalidate_sessions_for_deleted_data",
            new=AsyncMock(),
        ),
    ):
        await resolve._delete_dlt_orphans(
            "dataset",
            user,
            set(),
            sources=("google_drive",),
            document_scopes={("google_drive", "drive_a")},
        )
    delete.assert_awaited_once_with(owned, dataset.id)
