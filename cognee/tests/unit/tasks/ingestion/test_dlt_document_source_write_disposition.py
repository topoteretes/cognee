"""A document source's declared ``merge`` must survive a caller that does not repeat it.

Every connector declares ``write_disposition="merge"`` on its resource. An
incremental run yields only what changed, so with ``replace`` the staging table
is rewritten with those rows alone: the read-back no longer has the others, and
the orphan cleanup that follows deletes their documents.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")


async def _sync(rows, **kwargs) -> list[str]:
    """One sync of a merge-declared document source; the external ids read back."""
    dlt = pytest.importorskip("dlt")

    @dlt.resource(
        name="pages",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def pages():
        yield from rows

    resource = pages()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "notion")
    setattr(resource, PIPELINE_SCOPE_ATTR, "pages")
    with (
        patch.object(
            resolve,
            "_stable_row_ids",
            new=AsyncMock(side_effect=lambda found, *_: [uuid4() for _ in found]),
        ),
        patch.object(resolve, "_refresh_row_structure", new=AsyncMock()),
    ):
        items, _ = await resolve.resolve_dlt_sources(
            resource,
            "brain",
            SimpleNamespace(id=uuid4()),
            dataset_id=uuid4(),
            primary_key="id",
            **kwargs,
        )
    return sorted(item.system_metadata["external_id"] for item in items)


def _page(page_id: str, content: str) -> dict:
    return {"id": page_id, "title": page_id, "content": content, "_deleted": False}


@pytest.mark.asyncio
async def test_explicit_merge_keeps_the_rows_an_incremental_run_did_not_send(staging):
    await _sync([_page("a", "A"), _page("b", "B")], write_disposition="merge")

    after_edit = await _sync([_page("b", "B edited")], write_disposition="merge")

    assert after_edit == ["a", "b"]


@pytest.mark.asyncio
async def test_a_document_source_keeps_its_declared_merge_when_the_caller_omits_it(staging):
    await _sync([_page("a", "A"), _page("b", "B")])

    after_edit = await _sync([_page("b", "B edited")])

    # With the engine's "replace" default the staging table holds only "b", so "a"
    # is not read back and the orphan cleanup that follows deletes its document.
    assert after_edit == ["a", "b"]
