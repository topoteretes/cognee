"""Copy one document's graph, vectors and provenance from one dataset to another.

No LLM call and no embedding call: the destination gets the exact nodes, edges and
stored vectors the source holds for the document, so recall there behaves as it does
in the source. Elements the destination already holds merge by id (entity ids derive
from names, chunk ids from content): ``add_nodes`` / ``add_edges`` upsert, and the raw
vector upsert unions ``belongs_to_set``.

What is copied, and how it is found:

- Graph nodes and edges: every element whose source refs name the document in the
  source dataset. They are written to the destination with those refs rewritten to the
  destination dataset and its own ``Data`` row, so ``forget()`` there removes exactly
  them and leaves what other documents share.
- Vectors: the rows of those node ids in every collection of the source store, read
  with their stored vectors and written raw under the same collection names.
- The ``Data`` row: a new row in the destination that keeps the source row's id as
  ``legacy_id`` (the shape the per-dataset split gives forked rows, so either id
  resolves there), points at the same stored file, and is stamped cognified for the
  destination so the next cognify of that dataset does not extract it again. A second
  copy of the same document finds that row and reuses it.
- Edge evidence rows (which chunk supports which edge), re-keyed to the destination.

Only adapters that keep source refs on the graph can answer the lookup (Ladybug,
Neo4j, the Postgres demo); the relational-ledger backends raise
``UnsupportedProvenanceCapability``. The source dataset's edge list is read once to
pick the document's edges, because the graph interface has no edge lookup by identity.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

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
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME

from .resolve_data_id import resolve_data_id

# Graph provenance columns (``GraphDBInterface.attach_*_source_refs``). They describe
# the source dataset's ownership and are rebuilt for the destination, so they never
# travel as plain properties.
_PROVENANCE_FIELDS = ("source_ref_keys", "source_dataset_ids", "source_run_ids", "source_run_refs")


@dataclass(frozen=True)
class CopiedDocument:
    """What ``copy_data_to_dataset`` wrote: the destination's row and the element counts."""

    data_id: UUID
    data_record_created: bool
    nodes: int
    edges: int
    vectors: int
    evidence_rows: int


@dataclass(frozen=True)
class _DocumentSnapshot:
    """Everything the source dataset holds for one document."""

    nodes: list[dict[str, Any]]
    node_refs: dict[str, list[str]]
    edges: list[tuple[str, str, str, dict[str, Any]]]
    edge_refs: dict[EdgeIdentity, list[str]]
    vectors: dict[str, list[Any]]


async def copy_data_to_dataset(
    source_dataset: Dataset, data: Data, target_dataset: Dataset, user: User
) -> CopiedDocument:
    """Copy ``data`` from ``source_dataset`` into ``target_dataset``.

    Authorization is the caller's job (``datasets.copy_data`` checks read on the source
    and write on the destination); both dataset contexts are entered under their owner,
    as every dataset operation does. Entering a context checks the dataset's recorded
    embedding width against the configured model, so two datasets this can enter share
    one width, and the raw upsert checks every vector against it again.

    Raises:
        DataNotInGraphError: the source graph owns nothing for the document (never
            cognified, or a graph from before source refs existed).
        UnsupportedProvenanceCapability: the source graph adapter keeps no source refs.
    """
    snapshot = await _read_document(source_dataset, data.id)
    if not snapshot.nodes:
        raise DataNotInGraphError(data.id, source_dataset.id)

    target_data, created = await _ensure_target_data_row(data, target_dataset, user)
    nodes, edges, vectors = await _write_document(target_dataset, target_data.id, snapshot)
    evidence_rows = await _copy_edge_evidence(
        source_dataset.id, data.id, target_dataset.id, target_data.id, user
    )
    return CopiedDocument(
        data_id=target_data.id,
        data_record_created=created,
        nodes=nodes,
        edges=edges,
        vectors=vectors,
        evidence_rows=evidence_rows,
    )


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


async def _read_document(dataset: Dataset, data_id: UUID) -> _DocumentSnapshot:
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        node_refs = _refs_of_document(
            await graph.find_node_source_refs_by_dataset(str(dataset.id)), data_id
        )
        if not node_refs:
            return _DocumentSnapshot([], {}, [], {}, {})
        edge_refs = _refs_of_document(
            await graph.find_edge_source_refs_by_dataset(str(dataset.id)), data_id
        )
        node_ids = list(node_refs)
        nodes = [_without_provenance(node) for node in await graph.get_nodes(node_ids)]
        _, all_edges = await graph.get_graph_data()
        edges = [
            (source_id, target_id, relationship, _without_provenance(properties or {}))
            for source_id, target_id, relationship, properties in all_edges
            if EdgeIdentity(str(source_id), str(target_id), str(relationship)) in edge_refs
        ]

        vector_engine = await get_vector_engine_async()
        vectors: dict[str, list[Any]] = {}
        for name in await _collection_names(vector_engine):
            rows = await vector_engine.retrieve(name, node_ids, include_vector=True)
            if rows:
                vectors[name] = rows
    return _DocumentSnapshot(nodes, node_refs, edges, edge_refs, vectors)


async def _ensure_target_data_row(data: Data, target: Dataset, user: User) -> tuple[Data, bool]:
    """The destination's row for ``data``: the one an earlier copy left, or a new one.

    The new row is the source row with a new id, ``legacy_id`` naming the source id,
    owned by the user who copies, and stamped cognified for the destination.
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


async def _write_document(
    target: Dataset, target_data_id: UUID, snapshot: _DocumentSnapshot
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
    source_dataset_id: UUID,
    source_data_id: UUID,
    target_dataset_id: UUID,
    target_data_id: UUID,
    user: User,
) -> int:
    """Re-key the document's chunk -> edge support rows to the destination.

    The rows carry no pipeline run (none ran), which the evidence lookup treats as
    observed when written. Ids derive from the destination keys, so source rows that
    differed only by run collapse into one, and a repeated copy inserts nothing new.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        source_rows = (
            (
                await session.execute(
                    select(ProvenanceEdgeEvidence).where(
                        ProvenanceEdgeEvidence.dataset_id == source_dataset_id,
                        ProvenanceEdgeEvidence.data_id == source_data_id,
                    )
                )
            )
            .scalars()
            .all()
        )
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
