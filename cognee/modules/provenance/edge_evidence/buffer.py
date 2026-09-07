"""SQL-free, per-data-item buffering for edge evidence."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.databases.provenance import data_item_id
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.utils import generate_edge_object_id


def _as_uuid(value: Any) -> Optional[UUID]:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _edge_parts(edge: Any) -> Optional[tuple[UUID, UUID, str, dict]]:
    source = getattr(edge, "source_id", None)
    target = getattr(edge, "target_id", None)
    relationship = getattr(edge, "relationship_name", None)
    properties = {}
    if source is None or target is None or not relationship:
        if not isinstance(edge, (tuple, list)) or len(edge) < 3:
            return None
        source, target, relationship = edge[:3]
        properties = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}

    source_uuid = _as_uuid(source)
    target_uuid = _as_uuid(target)
    if source_uuid is None or target_uuid is None:
        return None
    return source_uuid, target_uuid, str(relationship), properties


def _evidence_id(
    *,
    tenant_id: Optional[UUID],
    user_id: UUID,
    dataset_id: UUID,
    data_id: UUID,
    pipeline_run_id: Optional[UUID],
    chunk_id: UUID,
    edge_id: UUID,
    evidence_kind: str,
) -> UUID:
    parts = (
        tenant_id or "",
        user_id,
        dataset_id,
        data_id,
        pipeline_run_id or "",
        chunk_id,
        edge_id,
        evidence_kind,
    )
    encoded = "\x1f".join(str(part) for part in parts)
    return uuid5(NAMESPACE_URL, f"cognee:edge-evidence:v1:{encoded}")


@dataclass(frozen=True, slots=True)
class ProvenanceBatch:
    evidence_rows: tuple[dict, ...]


@dataclass(slots=True)
class ProvenanceBuffer:
    """Deduplicated evidence rows for one data-item pipeline context."""

    evidence_rows: dict[UUID, dict] = field(default_factory=dict)

    def pending_record_count(self) -> int:
        return len(self.evidence_rows)

    def snapshot(self) -> ProvenanceBatch:
        return ProvenanceBatch(
            evidence_rows=tuple(dict(row) for row in self.evidence_rows.values())
        )

    def mark_persisted(self, batch: ProvenanceBatch) -> None:
        for row in batch.evidence_rows:
            self.evidence_rows.pop(row["id"], None)

    def capture(
        self,
        *,
        chunks: Iterable[DocumentChunk],
        graph_edges: Iterable[Any],
        ctx: Any,
    ) -> int:
        """Capture every chunk→edge support link without storage access."""
        dataset_id = _as_uuid(getattr(getattr(ctx, "dataset", None), "id", None))
        resolved_data_id = _as_uuid(data_item_id(getattr(ctx, "data_item", None)))
        user = getattr(ctx, "user", None)
        user_id = _as_uuid(getattr(user, "id", None))
        tenant_id = _as_uuid(getattr(user, "tenant_id", None))
        pipeline_run_id = _as_uuid(getattr(ctx, "pipeline_run_id", None))
        if dataset_id is None or resolved_data_id is None or user_id is None:
            return 0

        structural_by_chunk: dict[str, list[Any]] = {}
        for edge in graph_edges:
            parts = _edge_parts(edge)
            if parts is not None:
                structural_by_chunk.setdefault(str(parts[0]), []).append(edge)

        captured = 0
        now = datetime.now(timezone.utc)
        for chunk in chunks:
            chunk_id = _as_uuid(chunk.id)
            if chunk_id is None:
                continue
            candidates = [
                *(("extracted", edge) for edge in getattr(chunk, "_provenance_edges", [])),
                *(("structural", edge) for edge in structural_by_chunk.get(str(chunk_id), [])),
            ]
            for evidence_kind, edge in candidates:
                parts = _edge_parts(edge)
                if parts is None:
                    continue
                source_id, destination_id, relationship_name, properties = parts
                edge_id = UUID(
                    generate_edge_object_id(source_id, destination_id, relationship_name)
                )
                row_id = _evidence_id(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    data_id=resolved_data_id,
                    pipeline_run_id=pipeline_run_id,
                    chunk_id=chunk_id,
                    edge_id=edge_id,
                    evidence_kind=evidence_kind,
                )
                confidence = properties.get("confidence")
                if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                    confidence = None
                if row_id in self.evidence_rows:
                    continue
                self.evidence_rows[row_id] = {
                    "id": row_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "dataset_id": dataset_id,
                    "data_id": resolved_data_id,
                    "pipeline_run_id": pipeline_run_id,
                    "chunk_id": chunk_id,
                    "chunk_index": int(chunk.chunk_index),
                    "edge_id": edge_id,
                    "source_node_id": source_id,
                    "destination_node_id": destination_id,
                    "relationship_name": relationship_name,
                    "evidence_kind": evidence_kind,
                    "source_task": getattr(chunk, "source_task", None),
                    "confidence": confidence,
                    "created_at": now,
                }
                captured += 1
        return captured
