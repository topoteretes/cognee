"""Writing a `Data` row's new content, and reading/stamping its cognify status.

These live beside the package's other ``Data`` accessors rather than in the
update API module: the row is this module's table, and a second writer for it
in the API layer is a place its owners would not think to look. ``StagedContent``
travels with them — leaving the type in ``api/v1/update`` would make this module
import from the API layer, which is the dependency direction the move exists to
correct.
"""

import json
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data

# The cognify pipeline's status key: an incremental update produces the same
# graph state cognify would, so it stamps the same slot — that is what stops a
# later cognify() redoing the document.
COGNIFY_PIPELINE_NAME = "cognify_pipeline"


def _completed_status():
    """DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED, imported on use.

    The enum lives under ``cognee.modules.pipelines``, and the pipeline layer
    imports THIS package — so importing it at module scope closes a cycle that
    leaves ``data.methods`` half-initialised, at which point ``from
    cognee.modules.data.methods import <helper>`` silently binds the submodule
    instead of the function. Deferring the import keeps the dependency
    one-directional at import time.
    """
    from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

    return DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED


class StagedContent(BaseModel):
    """The new content, processed and stored — with the Data row untouched.

    Content-addressed storage writes are safe to make before anything is
    decided: until ``publish_updated_data`` flips the row, readers keep
    resolving the old file, and an abandoned staged file is inert garbage.
    """

    name: str
    raw_data_location: str
    original_data_location: str
    extension: str
    mime_type: str
    original_extension: str
    original_mime_type: str
    loader_engine: str
    content_hash: str
    raw_content_hash: str
    data_size: int


def merged_external_metadata(data: Data, node_set: Optional[List[str]]) -> dict:
    """The row's external metadata with an explicitly supplied node_set applied.

    Mirrors ``ingest_data``'s ``ext_metadata["node_set"] = node_set`` so an
    update tags new chunks with the node_set the CALLER passed rather than the
    one stored before it.
    """
    metadata = dict(data.external_metadata or {})
    if node_set:
        metadata["node_set"] = node_set
    return metadata


async def publish_updated_data(
    data_id: UUID,
    dataset_id: UUID,
    staged: StagedContent,
    token_count: int,
    node_set: Optional[List[str]],
) -> None:
    """The one-transaction publish: content, metadata, and status flip together.

    Everything readers can observe about the document — stored text location,
    hashes, size, token count, and the cognify-completed stamp — commits
    atomically. Any crash before this leaves the row entirely on the old
    version.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        if data_point is None:
            raise ValueError(f"data row {data_id} disappeared before publish")

        data_point.name = staged.name
        data_point.raw_data_location = staged.raw_data_location
        data_point.original_data_location = staged.original_data_location
        data_point.extension = staged.extension
        data_point.mime_type = staged.mime_type
        data_point.original_extension = staged.original_extension
        data_point.original_mime_type = staged.original_mime_type
        data_point.loader_engine = staged.loader_engine
        data_point.content_hash = staged.content_hash
        data_point.raw_content_hash = staged.raw_content_hash
        data_point.data_size = staged.data_size
        data_point.token_count = token_count
        if node_set:
            # Both copies move together: the column readers filter on, and the
            # metadata a later full cognify rebuilds its NodeSet tags from.
            data_point.node_set = json.dumps(node_set)
            data_point.external_metadata = merged_external_metadata(data_point, node_set)

        status_for_pipeline = data_point.pipeline_status.setdefault(COGNIFY_PIPELINE_NAME, {})
        status_for_pipeline[str(dataset_id)] = _completed_status()

        await session.merge(data_point)
        await session.commit()


async def mark_data_processed(data_id: UUID, dataset_id: UUID) -> None:
    """Stamp cognify completion so a later cognify() doesn't redo the document."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        if data_point is None:
            return
        status_for_pipeline = data_point.pipeline_status.setdefault(COGNIFY_PIPELINE_NAME, {})
        status_for_pipeline[str(dataset_id)] = _completed_status()
        await session.merge(data_point)
        await session.commit()


async def is_data_processed(data_id: UUID, dataset_id: UUID) -> bool:
    """Whether the cognify-completion stamp is already on the row."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        if data_point is None:
            return False
        status = (data_point.pipeline_status or {}).get(COGNIFY_PIPELINE_NAME, {})
        return status.get(str(dataset_id)) == _completed_status()
