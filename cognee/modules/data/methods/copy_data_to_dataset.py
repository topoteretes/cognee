"""Copy one document's graph, vectors and provenance from one dataset to another.

No LLM call and no embedding call: the destination gets the exact nodes, edges and
stored vectors the source holds for the document, so recall there behaves as it does
in the source. Elements the destination already holds merge by id (entity ids derive
from names, chunk ids from content): ``add_nodes`` / ``add_edges`` upsert, and the raw
vector upsert unions ``belongs_to_set``.

Two steps, so a caller can read the source without holding the destination's lock:

- ``read_document_snapshot`` collects everything the source dataset holds for the
  document: the nodes and edges whose source refs name it (``find_*_source_refs_by_
  dataset`` filtered to the document, then ``get_nodes`` / ``get_edge_delete_data`` for
  their properties), the rows of those node ids in every collection of the source
  store with their stored vectors, and the document's confirmed edge-evidence rows.
- ``write_document_copy`` gives the destination a ``Data`` row of its own (new id, the
  source row's id kept as ``legacy_id`` so either id resolves there, the same stored
  file, stamped cognified for the destination so the next cognify does not extract
  it again; a second copy finds and reuses it), then writes the nodes and edges with
  their refs re-addressed to that row, the vectors under the same collection names,
  and the evidence re-keyed. When a write fails and this copy created the row,
  everything written for it is removed again through the provenance delete
  ``forget()`` uses, so a failed first copy leaves the destination as it was.

Only adapters that keep source refs on the graph can answer the lookups (Ladybug,
Neo4j, the Postgres demo); the relational-ledger backends raise
``UnsupportedProvenanceCapability``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.source_refs import (
    make_chunk_source_ref_key,
    make_source_ref_key,
    parse_source_ref_key,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.data.exceptions import DataNotInGraphError, VectorAdapterIndexSchemaError
from cognee.modules.data.models import Data, Dataset
from cognee.modules.provenance.edge_evidence.buffer import ProvenanceBatch, _evidence_id
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence
from cognee.modules.provenance.edge_evidence.persistence import persist_provenance_batch
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME

from .delete_data import delete_data
from .resolve_data_id import resolve_data_id

logger = get_logger()

# Graph provenance columns (``GraphDBInterface.attach_*_source_refs``). They describe
# the source dataset's ownership and are rebuilt for the destination, so they never
# travel as plain properties.
_PROVENANCE_FIELDS = ("source_ref_keys", "source_dataset_ids", "source_run_ids", "source_run_refs")


@dataclass(frozen=True)
class DocumentSnapshot:
    """Everything the source dataset holds for one document."""

    source_dataset_id: UUID
    data_id: UUID
    nodes: list[dict[str, Any]]
    node_refs: dict[str, list[str]]
    edges: list[tuple[str, str, str, dict[str, Any]]]
    edge_refs: dict[EdgeIdentity, list[str]]
    vectors: dict[str, list[Any]]
    evidence: list[ProvenanceEdgeEvidence]


@dataclass(frozen=True)
class CopiedDocument:
    """What ``write_document_copy`` wrote: the destination's row and the element counts."""

    data_id: UUID
    data_record_created: bool
    nodes: int
    edges: int
    vectors: int
    evidence_rows: int


def _refs_of_document(refs: dict[Any, list[str]], data_id: UUID) -> dict[Any, list[str]]:
    """The entries of ``refs`` (element -> source ref keys) that name ``data_id``, kept
    to those keys. A node shared with another document keeps only this document's refs,
    which is what the destination may claim for it."""
    selected: dict[Any, list[str]] = {}
    for element, keys in refs.items():
        own = [key for key in keys if parse_source_ref_key(key).data_id == data_id]
        if own:
            selected[element] = own
    return selected


def _rewrite_ref(key: str, target_dataset_id: UUID, target_data_id: UUID) -> str:
    """The same ref (whole document or one chunk) addressed to the destination row."""
    parsed = parse_source_ref_key(key)
    if parsed.chunk_id is None:
        return make_source_ref_key(target_dataset_id, target_data_id)
    return make_chunk_source_ref_key(target_dataset_id, target_data_id, parsed.chunk_id)


def _without_provenance(properties: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in properties.items() if key not in _PROVENANCE_FIELDS}


async def _collection_names(vector_engine) -> list[str]:
    """Every collection of the current vector store.

    The interface has no listing call; pgvector and Turso expose ``get_table_names``
    (pgvector schema-qualified, with the graph demo's tables next to them), LanceDB its
    connection.
    """
    if hasattr(vector_engine, "get_table_names"):
        names = [name.split(".", 1)[-1] for name in await vector_engine.get_table_names()]
        return [name for name in names if not name.startswith("graph_")]
    connection = await vector_engine.get_connection()
    return list(await connection.table_names())


def _index_schema_of(vector_engine) -> type:
    """The row model the destination adapter stores index rows as.

    Every in-tree adapter writes its collections through its own ``IndexSchema``
    (``index_data_points``: id, the embedded text and the reference scalars such as
    ``document_id`` and ``chunk_index``), and its raw upsert validates payloads against
    the schema it is handed. Copied rows are validated against that same model, so the
    destination keeps every field the source row carried.
    """
    adapter_class = vector_engine.__class__
    schema = getattr(sys.modules[adapter_class.__module__], "IndexSchema", None)
    if schema is None:
        raise VectorAdapterIndexSchemaError(adapter_class.__name__)
    return schema


async def _confirmed_evidence(dataset_id: UUID, data_id: UUID) -> list[ProvenanceEdgeEvidence]:
    """The document's evidence rows the source itself counts: rows without a run, and
    rows whose run completed. The lookup (``_active_support``) ignores the rest, and the
    copy writes rows without a run, which it counts, so unconfirmed rows must not travel."""
    # Imported on use: the pipelines package imports this package's methods.
    from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(ProvenanceEdgeEvidence)
            .outerjoin(
                PipelineRun,
                and_(
                    PipelineRun.pipeline_run_id == ProvenanceEdgeEvidence.pipeline_run_id,
                    PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                ),
            )
            .where(
                ProvenanceEdgeEvidence.dataset_id == dataset_id,
                ProvenanceEdgeEvidence.data_id == data_id,
                or_(ProvenanceEdgeEvidence.pipeline_run_id.is_(None), PipelineRun.id.is_not(None)),
            )
        )
        return list(rows.scalars().all())


async def read_document_snapshot(dataset: Dataset, data: Data) -> DocumentSnapshot:
    """Everything ``dataset`` holds for ``data``, read under the dataset's owner.

    Raises:
        DataNotInGraphError: the graph owns nothing for the document (never cognified,
            or a graph from before source refs existed).
        UnsupportedProvenanceCapability: the graph adapter keeps no source refs.
    """
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        node_refs = _refs_of_document(
            await graph.find_node_source_refs_by_dataset(str(dataset.id)), data.id
        )
        if not node_refs:
            raise DataNotInGraphError(data.id, dataset.id)
        edge_refs = _refs_of_document(
            await graph.find_edge_source_refs_by_dataset(str(dataset.id)), data.id
        )
        node_ids = list(node_refs)
        nodes = [_without_provenance(node) for node in await graph.get_nodes(node_ids)]
        edge_data = await graph.get_edge_delete_data(list(edge_refs))
        edges = [
            (
                identity.source_id,
                identity.target_id,
                identity.relationship_name,
                _without_provenance(stored.edge_properties),
            )
            for identity, stored in edge_data.items()
        ]

        vector_engine = await get_vector_engine_async()
        vectors: dict[str, list[Any]] = {}
        for name in await _collection_names(vector_engine):
            rows = await vector_engine.retrieve(name, node_ids, include_vector=True)
            if rows:
                vectors[name] = rows
    evidence = await _confirmed_evidence(dataset.id, data.id)
    return DocumentSnapshot(
        dataset.id, data.id, nodes, node_refs, edges, edge_refs, vectors, evidence
    )


async def write_document_copy(
    target: Dataset, data: Data, snapshot: DocumentSnapshot, user: User
) -> CopiedDocument:
    """Write ``snapshot`` of ``data`` into ``target`` and return what was written.

    Call it under ``target``'s dataset lock. Entering the destination context checks its
    recorded embedding width against the configured model, as reading the source did,
    so the two share one width; the raw upsert checks every vector against it again.
    """
    target_data, created = await _ensure_target_data_row(data, target, user)
    try:
        nodes, edges, vectors = await _write_graph_and_vectors(target, target_data.id, snapshot)
        evidence_rows = await _copy_edge_evidence(
            snapshot.evidence, target.id, target_data.id, user
        )
    except BaseException:
        if created:
            # A row stamped cognified with a partial graph behind it would be skipped
            # by every later cognify and never repaired: take the copy back out.
            await _discard_copy(target, target_data)
        raise
    return CopiedDocument(
        data_id=target_data.id,
        data_record_created=created,
        nodes=nodes,
        edges=edges,
        vectors=vectors,
        evidence_rows=evidence_rows,
    )


async def _ensure_target_data_row(data: Data, target: Dataset, user: User) -> tuple[Data, bool]:
    """The destination's row for ``data``: the one an earlier copy left, or a new one.

    The new row is the source row with a new id, ``legacy_id`` naming the source id,
    owned by the user who copies, and stamped cognified for the destination. The stamp
    is safe to write first because ``write_document_copy`` discards the row again when
    a later write fails.
    """
    # Imported on use: the pipelines package imports this package's methods.
    from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        existing_id = await resolve_data_id(target.id, data.id)
        if existing_id is not None:
            return await session.get(Data, existing_id), False

        # TODO(SDK-1435): a destination that ingested the same content on its own is
        # not detected. That row has its own id, and chunk ids embed the document id,
        # so this copy adds a second Document node and a second set of chunks (with
        # their edges and evidence) for the same text. Refusing on a matching
        # ``content_hash`` (``get_dataset_data_by_content_hash``, the predicate
        # ingestion dedup uses) covers file-backed rows only: not every data type
        # carries a content hash, so the check needs a per-type answer before it can
        # be the rule. Until then, callers copying into a dataset that may already
        # hold the document should look for it first.

        row = Data(
            id=uuid4(),
            legacy_id=data.id,
            dataset_id=target.id,
            owner_id=user.id,
            tenant_id=user.tenant_id if user.tenant_id else None,
            name=data.name,
            label=data.label,
            extension=data.extension,
            mime_type=data.mime_type,
            original_extension=data.original_extension,
            original_mime_type=data.original_mime_type,
            loader_engine=data.loader_engine,
            raw_data_location=data.raw_data_location,
            original_data_location=data.original_data_location,
            content_hash=data.content_hash,
            raw_content_hash=data.raw_content_hash,
            external_metadata=data.external_metadata,
            system_metadata=data.system_metadata,
            node_set=data.node_set,
            token_count=data.token_count,
            data_size=data.data_size,
            importance_weight=data.importance_weight,
            pipeline_status={
                COGNIFY_PIPELINE_NAME: {
                    str(target.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
                }
            },
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row, True


async def _attach_grouped(attach, keyed_refs: dict[Any, list[str]]) -> None:
    """``attach_*_source_refs`` takes one key list for many elements: group the elements
    by identical key list so each distinct list is one call."""
    groups: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for element, keys in keyed_refs.items():
        groups[tuple(sorted(keys))].append(element)
    for keys, elements in groups.items():
        await attach(elements, list(keys), None)


async def _write_graph_and_vectors(
    target: Dataset, target_data_id: UUID, snapshot: DocumentSnapshot
) -> tuple[int, int, int]:
    node_refs = {
        node_id: [_rewrite_ref(key, target.id, target_data_id) for key in keys]
        for node_id, keys in snapshot.node_refs.items()
    }
    edge_refs = {
        identity: [_rewrite_ref(key, target.id, target_data_id) for key in keys]
        for identity, keys in snapshot.edge_refs.items()
    }

    async with set_database_global_context_variables(target.id, target.owner_id):
        graph = await get_graph_engine()
        node_objects = [SimpleNamespace(**properties) for properties in snapshot.nodes]
        if getattr(graph, "supports_per_row_source_refs", False):
            # The bulk writes stamp one ref per row; the rest of a row's refs follow.
            await graph.add_nodes(
                node_objects, source_ref_key={nid: keys[0] for nid, keys in node_refs.items()}
            )
            await _attach_grouped(
                graph.attach_node_source_refs,
                {nid: keys[1:] for nid, keys in node_refs.items() if len(keys) > 1},
            )
            await graph.add_edges(
                snapshot.edges,
                source_ref_key={
                    (identity.source_id, identity.target_id, identity.relationship_name): keys[0]
                    for identity, keys in edge_refs.items()
                },
            )
            await _attach_grouped(
                graph.attach_edge_source_refs,
                {identity: keys[1:] for identity, keys in edge_refs.items() if len(keys) > 1},
            )
        else:
            await graph.add_nodes(node_objects)
            await _attach_grouped(graph.attach_node_source_refs, node_refs)
            await graph.add_edges(snapshot.edges)
            await _attach_grouped(graph.attach_edge_source_refs, edge_refs)

        vector_engine = await get_vector_engine_async()
        index_schema = _index_schema_of(vector_engine)
        written = 0
        for name, rows in snapshot.vectors.items():
            points = []
            for row in rows:
                payload = dict(row.payload)
                vector = payload.pop("vector")
                points.append({"id": str(row.id), "vector": list(vector), "payload": payload})
            await vector_engine.upsert_raw_vectors(name, points, payload_schema=index_schema)
            written += len(points)

    return len(snapshot.nodes), len(snapshot.edges), written


async def _copy_edge_evidence(
    source_rows: list[ProvenanceEdgeEvidence],
    target_dataset_id: UUID,
    target_data_id: UUID,
    user: User,
) -> int:
    """Re-key the document's chunk -> edge support rows to the destination.

    The rows carry no pipeline run (none ran), which the evidence lookup treats as
    observed when written. Ids derive from the destination keys, so source rows that
    differed only by run collapse into one, and a repeated copy inserts nothing new.
    """
    if not source_rows:
        return 0

    tenant_id = user.tenant_id if user.tenant_id else None
    now = datetime.now(timezone.utc)
    rows_by_id = {
        row["id"]: row
        for row in (
            {
                "id": _evidence_id(
                    tenant_id=tenant_id,
                    user_id=user.id,
                    dataset_id=target_dataset_id,
                    data_id=target_data_id,
                    pipeline_run_id=None,
                    chunk_id=row.chunk_id,
                    edge_id=row.edge_id,
                    evidence_kind=row.evidence_kind,
                ),
                "tenant_id": tenant_id,
                "user_id": user.id,
                "dataset_id": target_dataset_id,
                "data_id": target_data_id,
                "pipeline_run_id": None,
                "chunk_id": row.chunk_id,
                "chunk_index": row.chunk_index,
                "edge_id": row.edge_id,
                "source_node_id": row.source_node_id,
                "destination_node_id": row.destination_node_id,
                "relationship_name": row.relationship_name,
                "evidence_kind": row.evidence_kind,
                "source_task": row.source_task,
                "confidence": row.confidence,
                "created_at": now,
            }
            for row in source_rows
        )
    }
    return await persist_provenance_batch(ProvenanceBatch(evidence_rows=tuple(rows_by_id.values())))


async def _discard_copy(target: Dataset, target_data: Data) -> None:
    """Undo a failed first copy: drop the row's refs from the destination graph (which
    hard-deletes what nothing else owns, vectors included), then its evidence rows and
    the row itself. Elements other documents share keep their own refs. A failure here
    is logged, not raised: the write error the caller is already propagating is the one
    to report, and the row left behind is what ``forget()`` removes."""
    from cognee.infrastructure.databases.unified import get_unified_engine

    try:
        async with set_database_global_context_variables(target.id, target.owner_id):
            unified = await get_unified_engine()
            await unified.delete_by_document(str(target.id), str(target_data.id))
            await delete_data([target_data], target.id)
    except Exception:
        logger.exception(
            "Could not discard the failed copy %s in dataset %s; forget it there by hand",
            target_data.id,
            target.id,
        )
