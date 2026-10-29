"""Orphan cleanup must act on the dataset the rows were written to (#5983).

A dataset name resolves only among the caller's own datasets, so when add()
writes a dlt source to a shared dataset by id, a name lookup either picks the
caller's own same-named dataset (and sweeps its rows) or finds nothing. The
cleanup therefore resolves by id whenever one is given.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")
# The package re-exports get_dataset_data under the same name as its submodule, so
# a dotted patch() target resolves to the function on Python 3.10 (mock walks the
# path with getattr there); import the submodule itself and patch that.
get_dataset_data_module = importlib.import_module("cognee.modules.data.methods.get_dataset_data")


@pytest.mark.asyncio
async def test_orphan_cleanup_resolves_dataset_by_id_when_given():
    user = SimpleNamespace(id=uuid4())
    shared_dataset_id = uuid4()

    with patch(
        "cognee.modules.data.methods.get_authorized_existing_datasets",
        new=AsyncMock(return_value=[]),
    ) as lookup:
        await resolve._delete_dlt_orphans(
            "notes",
            user,
            set(),
            dataset_id=shared_dataset_id,
        )

    lookup.assert_awaited_once_with(
        user=user, permission_type="delete", datasets=[shared_dataset_id]
    )


@pytest.mark.asyncio
async def test_orphan_cleanup_falls_back_to_name_without_id():
    user = SimpleNamespace(id=uuid4())

    with patch(
        "cognee.modules.data.methods.get_authorized_existing_datasets",
        new=AsyncMock(return_value=[]),
    ) as lookup:
        await resolve._delete_dlt_orphans("notes", user, set())

    lookup.assert_awaited_once_with(user=user, permission_type="delete", datasets=["notes"])


@pytest.mark.asyncio
async def test_orphan_cleanup_skips_without_delete_permission():
    """A write-only grantee's ingest succeeds; the sweep is skipped, nothing is deleted."""
    from cognee.modules.users.exceptions import PermissionDeniedError

    user = SimpleNamespace(id=uuid4())

    with (
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            new=AsyncMock(side_effect=PermissionDeniedError("no delete")),
        ),
        patch.object(get_dataset_data_module, "get_dataset_data", new=AsyncMock()) as dataset_data,
    ):
        await resolve._delete_dlt_orphans("notes", user, set(), dataset_id=uuid4())

    dataset_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_dlt_sources_cleanup_passes_dataset_id_through():
    """Both cleanup calls scheduled by resolve_dlt_sources carry the dataset id."""
    dlt = pytest.importorskip("dlt")

    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

    @dlt.resource(name="pages", primary_key="id", write_disposition="replace")
    def pages():
        yield {"id": 1, "title": "one", "content": "page one"}

    @dlt.source(name="testdocs")
    def source():
        return pages

    dlt_source = source()
    setattr(dlt_source, DOCUMENT_SOURCE_ATTR, "testdocs")

    user = SimpleNamespace(id=uuid4())
    shared_dataset_id = uuid4()
    orphans = AsyncMock()

    with (
        patch.object(resolve, "_delete_dlt_orphans", new=orphans),
        patch.object(
            resolve,
            "_stable_row_ids",
            new=AsyncMock(side_effect=lambda rows, *_: [uuid4() for _ in rows]),
        ),
        patch.object(resolve, "_refresh_row_structure", new=AsyncMock()),
    ):
        _, cleanup = await resolve.resolve_dlt_sources(
            [dlt_source], dataset_name="notes", user=user, dataset_id=shared_dataset_id
        )
        assert cleanup is not None
        await cleanup()

    assert orphans.await_count >= 1
    for call in orphans.await_args_list:
        assert call.kwargs["dataset_id"] == shared_dataset_id
