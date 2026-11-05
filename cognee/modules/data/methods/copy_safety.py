"""Versioned copy receipts and preflight checks shared with Cloud approvals."""

import hashlib
import json

from sqlalchemy import select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.users.models import DatasetDatabase


def fingerprint(data, snapshot):
    """Bind approval to all content that travels, not just a file name or timestamp."""
    payload = {
        "source": str(snapshot.source_dataset_id),
        "document": str(data.id),
        "metadata": {
            key: getattr(data, key)
            for key in (
                "content_hash",
                "raw_content_hash",
                "name",
                "external_metadata",
                "system_metadata",
                "node_set",
                "raw_data_location",
                "original_data_location",
            )
        },
        "model": snapshot.embedding_identity,
        "nodes": sorted(snapshot.nodes, key=lambda row: str(row["id"])),
        "edges": sorted(snapshot.edges, key=lambda row: tuple(map(str, row[:3]))),
        "vectors": {
            name: [
                {"id": str(row.id), "payload": row.payload}
                for row in sorted(rows, key=lambda row: str(row.id))
            ]
            for name, rows in snapshot.vectors.items()
        },
        "evidence": sorted({str(row.id) for row in snapshot.evidence}),
    }
    return hashlib.sha256(json.dumps(payload, cls=JSONEncoder, sort_keys=True).encode()).hexdigest()


async def embedding_identity(dataset_id):
    async with get_relational_engine().get_async_session() as session:
        row = await session.get(DatasetDatabase, dataset_id)
        info = row.vector_database_connection_info if row else {}
    model, dimensions = info.get("embedding_model"), info.get("embedding_dimensions")
    if not model or not dimensions:
        raise CogneeValidationError(
            "Copying requires a recorded embedding model and dimensions. Rebuild legacy embeddings first.",
            status_code=409,
        )
    return model, dimensions


async def validate_destination_embedding(target, source_identity):
    # Enter first so a new empty destination records its configured embedding model.
    async with set_database_global_context_variables(target.id, target.owner_id):
        target_identity = await embedding_identity(target.id)
    if target_identity != source_identity:
        raise CogneeValidationError(
            "Source and destination must use the same embedding model and dimensions.",
            status_code=409,
        )


async def source_revision(dataset_id, data_id):
    """Detect mutations between the unlocked destination preflight and lock handoff."""
    from cognee.modules.data.models import Data
    from cognee.modules.pipelines.models import PipelineRun

    async with get_relational_engine().get_async_session() as session:
        data = await session.get(Data, data_id)
        if data is None or data.dataset_id != dataset_id:
            raise CogneeValidationError(
                "Source document was removed. Refresh before copying.", status_code=409
            )
        run = await session.scalar(
            select(PipelineRun.id)
            .where(
                PipelineRun.dataset_id == dataset_id,
            )
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
            .limit(1)
        )
        return data.updated_at, data.content_hash, data.raw_content_hash, run
