"""Document orphan cleanup requires evidence for the exact staging table.

The sync side of this (a table emptied by tombstones, a source that staged nothing) runs on
the real stack in tasks/ingestion/test_dlt_document_sync.py.
"""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.tasks.ingestion.dlt_row_data import DltRows
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")


async def _sweep(stored_rows, fresh_ids, **options):
    """Runs the real ``_delete_dlt_orphans`` over ``stored_rows``; returns what it deleted."""
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())

    @asynccontextmanager
    async def context(*args):
        yield

    with (
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch.object(
            importlib.import_module("cognee.modules.data.methods.get_dataset_data"),
            "get_dataset_data",
            new=AsyncMock(return_value=stored_rows),
        ),
        patch("cognee.context_global_variables.set_database_global_context_variables", context),
        patch.object(
            importlib.import_module("cognee.modules.graph.methods.delete_data_nodes_and_edges"),
            "delete_data_nodes_and_edges",
            new=AsyncMock(return_value=SimpleNamespace(node_ids=[], edge_ids=[])),
        ),
        patch(
            "cognee.modules.conflicts.methods.retire_forgotten_decisions",
            new=AsyncMock(return_value=set()),
        ),
        patch.object(
            importlib.import_module("cognee.modules.data.methods.delete_data"),
            "delete_data",
            new=AsyncMock(),
        ) as delete,
        patch.object(
            importlib.import_module("cognee.modules.session_lifecycle.invalidate_sessions"),
            "invalidate_sessions_for_deleted_data",
            new=AsyncMock(),
        ),
    ):
        await resolve._delete_dlt_orphans("dataset", user, fresh_ids, **options)
    return delete, dataset, user


@pytest.mark.asyncio
async def test_empty_table_cleanup_retains_other_tables_and_unscoped_legacy_rows():
    owned = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "google_drive", "table_name": "drive_a"}
    )
    other = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "google_drive", "table_name": "drive_b"}
    )
    legacy = SimpleNamespace(id=uuid4(), system_metadata={"source": "google_drive"})

    delete, dataset, user = await _sweep(
        [owned, other, legacy],
        set(),
        sources=("google_drive",),
        document_scopes={("google_drive", "drive_a")},
    )

    delete.assert_awaited_once_with([owned], dataset.id, user)


@pytest.mark.asyncio
async def test_a_document_the_run_kept_is_not_forgotten():
    """Another source of the same add may have just stored the very document this sync's
    sweep names as stale: ids derive from content, so both want it."""
    stale = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "google_drive", "table_name": "drive_a"}
    )
    kept = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "google_drive", "table_name": "drive_a"}
    )

    delete, dataset, user = await _sweep(
        [stale, kept],
        {kept.id},
        sources=("google_drive",),
        document_scopes={("google_drive", "drive_a")},
    )

    delete.assert_awaited_once_with([stale], dataset.id, user)
