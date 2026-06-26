"""Translate COGX records into Cognee ingestion inputs.

Two targets, selected by the source's import mode:

- **data items** (``re-derive``): textual records become :class:`DataItem`s
  with deterministic ``data_id``s, fed through the normal ``add + cognify``
  path so Cognee's own extraction builds the graph.
- **graph batches** (``preserve``/``hybrid``): entity and fact records become
  native :class:`Entity` DataPoints plus custom edge tuples, stored directly
  via the ``add_data_points`` task with zero LLM calls. Temporal validity from
  the source (``valid_at``/``invalid_at``) is preserved as edge properties.
"""

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Set, Tuple
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.migration.cogx import (
    COGXEntity,
    COGXEpisode,
    COGXFact,
    COGXRawNode,
    COGXRecord,
)
from cognee.modules.migration.snapshot import rehydrate_node
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

logger = get_logger("migration.loader")

FACTS_PER_DIGEST = 200
# Target node count per graph batch: keeps each add_data_points call (gather,
# dedup, deep copies, relational transaction) bounded on bulk imports.
BATCH_NODE_TARGET = 2000


@dataclass
class TranslationResult:
    data_items: List[DataItem] = field(default_factory=list)
    graph_batches: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    # Facts dropped because a subject/object UUID reference could not be
    # resolved to any exported node (never fabricated as UUID-named entities).
    skipped_facts: int = 0
    # False in preserve mode: data items are stored raw, without cognify.
    cognify_data_items: bool = True


def record_data_id(record: COGXRecord) -> UUID:
    """Deterministic data id so re-importing the same record is idempotent."""
    return uuid5(NAMESPACE_OID, f"cogx:{record.external_system}:{record.external_id}")


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _record_external_metadata(record: COGXRecord) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "external_system": record.external_system,
        "external_id": record.external_id,
    }
    scope = record.scope.model_dump(exclude_none=True)
    if scope:
        metadata["scope"] = scope
    if record.created_at:
        metadata["external_created_at"] = _iso(record.created_at)
    if record.updated_at:
        metadata["external_updated_at"] = _iso(record.updated_at)
    if record.metadata:
        metadata.update(record.metadata)
    return metadata


def render_episode(episode: COGXEpisode) -> str:
    """Render an episode as a timestamped transcript."""
    lines = []
    if episode.title:
        lines.append(f"# {episode.title}")
    turns = sorted(
        episode.turns,
        key=lambda turn: turn.occurred_at.timestamp() if turn.occurred_at else float("-inf"),
    )
    for turn in turns:
        timestamp = f" [{_iso(turn.occurred_at)}]" if turn.occurred_at else ""
        lines.append(f"{turn.role}{timestamp}: {turn.content}")
    return "\n".join(lines)


def _render_fact_line(fact: COGXFact) -> str:
    line = fact.fact_text or f"{fact.subject_ref} {fact.predicate} {fact.object_ref}"
    qualifiers = []
    if fact.valid_at:
        qualifiers.append(f"valid from {_iso(fact.valid_at)}")
    if fact.invalid_at:
        qualifiers.append(f"invalid since {_iso(fact.invalid_at)}")
    if qualifiers:
        line = f"{line} ({', '.join(qualifiers)})"
    return line


def _data_item_for(record: COGXRecord, content: str, label: Optional[str] = None) -> DataItem:
    return DataItem(
        data=content,
        label=label,
        external_metadata=_record_external_metadata(record),
        data_id=record_data_id(record),
    )


def _looks_like_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def data_item_from_record(record: COGXRecord) -> Optional[DataItem]:
    """Translate a content-bearing record into a DataItem; None for graph records."""
    if record.kind == "document":
        return _data_item_for(record, record.content, record.title)
    if record.kind == "episode":
        return _data_item_for(record, render_episode(record), record.title)
    if record.kind == "memory":
        content = record.content
        if record.categories:
            content = f"{content}\nCategories: {', '.join(record.categories)}"
        return _data_item_for(record, content)
    if record.kind == "memory_block":
        return _data_item_for(record, f"{record.label}:\n{record.value}", record.label)
    return None


def _fact_edge_properties(fact: COGXFact) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "relationship_name": fact.predicate,
        "source_system": fact.external_system,
        "source_external_id": fact.external_id,
    }
    if fact.fact_text:
        properties["edge_text"] = fact.fact_text
    if fact.valid_at:
        properties["valid_at"] = _iso(fact.valid_at)
    if fact.invalid_at:
        properties["invalid_at"] = _iso(fact.invalid_at)
    if fact.confidence is not None:
        properties["confidence"] = fact.confidence
    return properties


def _register_entity(
    record: COGXEntity,
    *,
    entity_types: Dict[str, EntityType],
    by_node_id: Dict[UUID, Any],
    by_external_id: Dict[str, Any],
    first_external_id: Dict[UUID, str],
) -> Any:
    """Register an entity record, merging same-named records into one node."""

    def _entity_type_for(name: str) -> EntityType:
        key = name.lower()
        if key not in entity_types:
            entity_types[key] = EntityType(id=generate_node_id(name), name=name, description=name)
        return entity_types[key]

    node_id = generate_node_id(record.name)
    description = record.description or record.name
    if record.aliases:
        description = f"{description} Also known as: {', '.join(record.aliases)}."
    existing = by_node_id.get(node_id)
    if existing is not None:
        # Same-named source records merge into one node: combine their
        # descriptions/aliases instead of keeping only the first.
        if description and description not in (getattr(existing, "description", "") or ""):
            merged = getattr(existing, "description", None)
            existing.description = f"{merged}\n{description}" if merged else description
        if getattr(existing, "is_a", None) is None and record.entity_type:
            existing.is_a = _entity_type_for(record.entity_type)
        logger.info(
            "Merged same-named entity %r: external_ids %r and %r",
            record.name,
            first_external_id.get(node_id),
            record.external_id,
        )
        by_external_id[record.external_id] = existing
        return existing
    entity = Entity(
        id=node_id,
        name=record.name,
        description=description,
        is_a=_entity_type_for(record.entity_type) if record.entity_type else None,
    )
    by_node_id[node_id] = entity
    first_external_id[node_id] = record.external_id
    by_external_id[record.external_id] = entity
    return entity


def _build_graph_batches(
    entities: List[COGXEntity], facts: List[COGXFact], raw_nodes: List[COGXRawNode]
) -> Tuple[List[Dict[str, Any]], int]:
    """Map entity/fact/raw-node records onto bounded graph batches.

    Entity ids come from ``generate_node_id(name)`` — the same scheme cognify
    uses — so preserved facts merge into the existing graph vocabulary instead
    of forming a disconnected parallel graph. Raw nodes are rehydrated back
    into DataPoint instances (keeping their original ids) so facts referencing
    them stay resolvable. Facts whose subject/object UUID reference cannot be
    resolved are skipped — never fabricated as UUID-named entities.

    Nodes are split into batches of ~``BATCH_NODE_TARGET``; each fact lands in
    a batch containing one of its endpoints, with the other endpoint included
    as a (deterministic-id, hence idempotent) duplicate where batches split.

    Returns the batches plus the count of skipped facts.
    """
    if not entities and not facts and not raw_nodes:
        return [], 0

    entity_types: Dict[str, EntityType] = {}
    by_external_id: Dict[str, Any] = {}
    by_node_id: Dict[UUID, Any] = {}
    first_external_id: Dict[UUID, str] = {}

    for record in raw_nodes:
        properties = record.properties or {}
        node = rehydrate_node(properties)
        node = by_node_id.setdefault(node.id, node)
        external_id = properties.get("id")
        if external_id:
            by_external_id[str(external_id)] = node

    for record in entities:
        _register_entity(
            record,
            entity_types=entity_types,
            by_node_id=by_node_id,
            by_external_id=by_external_id,
            first_external_id=first_external_id,
        )

    batches: List[Dict[str, Any]] = []
    batch_index_of: Dict[UUID, int] = {}
    ordered_nodes = list(entity_types.values()) + list(by_node_id.values())
    for start in range(0, len(ordered_nodes), BATCH_NODE_TARGET):
        chunk = ordered_nodes[start : start + BATCH_NODE_TARGET]
        batches.append({"nodes": list(chunk), "edges": []})
        for node in chunk:
            batch_index_of[node.id] = len(batches) - 1

    def _resolve_ref(ref: str) -> Optional[Any]:
        node = by_external_id.get(ref)
        if node is not None:
            return node
        node = by_node_id.get(generate_node_id(ref))
        if node is not None:
            return node
        if _looks_like_uuid(ref):
            # A UUID pointing at a node the archive does not contain: skip the
            # fact rather than fabricate an Entity literally named by a UUID.
            return None
        # Plain-name reference (cross-provider archives): treat it as an
        # entity name and create the entity.
        entity = Entity(id=generate_node_id(ref), name=ref, description=ref)
        by_node_id[entity.id] = entity
        return entity

    duplicated_in_batch: Dict[int, Set[UUID]] = {}
    skipped_facts = 0

    for fact in facts:
        subject = _resolve_ref(fact.subject_ref)
        target = _resolve_ref(fact.object_ref)
        if subject is None or target is None:
            skipped_facts += 1
            unresolved = [
                ref
                for ref, node in ((fact.subject_ref, subject), (fact.object_ref, target))
                if node is None
            ]
            logger.warning(
                "Skipping fact %r (%s): unresolved UUID reference(s) %s",
                fact.external_id,
                fact.predicate,
                ", ".join(unresolved),
            )
            continue

        index = batch_index_of.get(subject.id)
        if index is None:
            index = batch_index_of.get(target.id)
        if index is None:
            if not batches:
                batches.append({"nodes": [], "edges": []})
            index = len(batches) - 1
        for node in (subject, target):
            if node.id not in batch_index_of:
                # Newly created name-stub entity: place it with this fact.
                batches[index]["nodes"].append(node)
                batch_index_of[node.id] = index
            elif batch_index_of[node.id] != index:
                # Endpoint lives in another batch: include a duplicate so the
                # edge's batch is self-contained (deterministic ids merge).
                duplicated = duplicated_in_batch.setdefault(index, set())
                if node.id not in duplicated:
                    batches[index]["nodes"].append(node)
                    duplicated.add(node.id)

        batches[index]["edges"].append(
            (subject.id, target.id, fact.predicate, _fact_edge_properties(fact))
        )

    batches = [batch for batch in batches if batch["nodes"] or batch["edges"]]
    return batches, skipped_facts


class _RecordTranslator:
    """Incremental record translator shared by the sync and async entry points."""

    def __init__(self, mode: str):
        self.mode = mode
        self.result = TranslationResult(cognify_data_items=mode != "preserve")
        self.entities: List[COGXEntity] = []
        self.facts: List[COGXFact] = []
        self.raw_nodes: List[COGXRawNode] = []

    def add(self, record: COGXRecord) -> None:
        result = self.result
        result.counts[record.kind] = result.counts.get(record.kind, 0) + 1

        data_item = data_item_from_record(record)
        if data_item is not None:
            result.data_items.append(data_item)
        elif record.kind == "entity":
            self.entities.append(record)
        elif record.kind == "fact":
            self.facts.append(record)
        elif record.kind == "raw_node":
            # Graph-fidelity payload: only meaningful for preserve/hybrid.
            self.raw_nodes.append(record)

    def finish(self) -> TranslationResult:
        result, entities, facts = self.result, self.entities, self.facts
        if self.mode == "re-derive":
            # Render the source's derived knowledge as digest documents so it
            # is not lost, and let cognify re-extract it. Raw nodes carry no
            # standalone text and are intentionally dropped in this mode.
            described = [e for e in entities if e.description]
            if described:
                lines = [f"{e.name}: {e.description}" for e in described]
                digest = COGXEntity(
                    external_system=described[0].external_system,
                    external_id="entities-digest",
                    name="entities-digest",
                )
                result.data_items.append(
                    _data_item_for(digest, "\n".join(lines), "Imported entity descriptions")
                )
            for start in range(0, len(facts), FACTS_PER_DIGEST):
                chunk = facts[start : start + FACTS_PER_DIGEST]
                digest = COGXFact(
                    external_system=chunk[0].external_system,
                    external_id=f"facts-digest-{start // FACTS_PER_DIGEST}",
                    subject_ref="-",
                    predicate="-",
                    object_ref="-",
                )
                result.data_items.append(
                    _data_item_for(
                        digest,
                        "\n".join(_render_fact_line(fact) for fact in chunk),
                        "Imported facts",
                    )
                )
        else:
            batches, skipped_facts = _build_graph_batches(entities, facts, self.raw_nodes)
            result.graph_batches.extend(batches)
            result.skipped_facts = skipped_facts
        return result


def translate_records(records: Iterable[COGXRecord], mode: str) -> TranslationResult:
    """Translate COGX records according to the import fidelity mode."""
    translator = _RecordTranslator(mode)
    for record in records:
        translator.add(record)
    return translator.finish()


async def translate_record_stream(
    records: AsyncIterable[COGXRecord], mode: str
) -> TranslationResult:
    """Translate an async record stream without first materializing it as a list."""
    translator = _RecordTranslator(mode)
    async for record in records:
        translator.add(record)
    return translator.finish()


def wrap_graph_batch(batch: Dict[str, Any], source_system: str, index: int) -> DataItem:
    """Wrap a graph batch in a DataItem with a deterministic data_id.

    The pipeline runtime treats DataItems with a stable ``data_id`` as
    first-class data items: no file storage, idempotent re-run bookkeeping.
    """
    fingerprint = "|".join(
        sorted(str(node.id) for node in batch["nodes"])
        + sorted(f"{s}-{r}-{t}" for s, t, r, _ in batch["edges"])
    )
    return DataItem(
        data=batch,
        label=f"migration-graph-batch-{index}",
        external_metadata={"external_system": source_system, "kind": "graph_batch"},
        data_id=uuid5(NAMESPACE_OID, f"cogx-graph:{source_system}:{fingerprint}"),
    )


def _provenance_ctx(ctx):
    """Adapt the pipeline context for add_data_points.

    add_data_points stamps ledger provenance from ``ctx.data_item.id``, which
    exists on Data ORM records but not on raw ingestion DataItems. Substitute
    the DataItem's deterministic ``data_id`` so provenance still lands.
    """
    if ctx is None:
        return None
    data_item = getattr(ctx, "data_item", None)
    if data_item is None or hasattr(data_item, "id"):
        return ctx
    data_id = getattr(data_item, "data_id", None)
    return SimpleNamespace(
        user=getattr(ctx, "user", None),
        dataset=getattr(ctx, "dataset", None),
        data_item=SimpleNamespace(id=data_id) if data_id else None,
        pipeline_run_id=getattr(ctx, "pipeline_run_id", None),
        pipeline_name=getattr(ctx, "pipeline_name", None),
        extras=getattr(ctx, "extras", None),
        _provenance_visited=getattr(ctx, "_provenance_visited", set()),
        task_sequence=getattr(ctx, "task_sequence", []),
    )


# Edge batches can run larger than node batches: edges are small tuples and
# add_data_points' per-node work (model traversal, embedding) does not apply.
EDGE_BATCH_TARGET = 2 * BATCH_NODE_TARGET


async def stream_graph_from_source(source, stats: Dict[str, int], ctx=None) -> Dict[str, int]:
    """Two-pass streaming graph import for replayable preserve-mode sources.

    Pass 1 streams the records once: raw nodes are rehydrated and flushed to
    storage in bounded batches (only an id registry is kept), while entity
    records buffer until the pass ends so same-name merging stays complete,
    then flush in bounded batches too. Pass 2 streams the records again,
    resolving each fact against the slim id registry and flushing edge
    batches. Endpoint nodes are guaranteed to exist in the graph store by the
    time edges arrive, so edges reference node ids without re-shipping nodes.

    Peak memory is the entity set plus one in-flight batch — instead of every
    rehydrated node and fact in the archive. ``stats`` (graph_nodes,
    graph_edges, skipped_facts) is mutated in place so the caller can report
    progress even when this task runs inside a background pipeline.
    """
    from cognee.tasks.storage.add_data_points import add_data_points

    ctx = _provenance_ctx(ctx)

    entity_types: Dict[str, EntityType] = {}
    by_external_id: Dict[str, Any] = {}
    by_node_id: Dict[UUID, Any] = {}
    first_external_id: Dict[UUID, str] = {}
    known_node_ids: Set[UUID] = set()
    external_to_node_id: Dict[str, UUID] = {}

    async def flush(nodes: List[Any], edges: Optional[List[Tuple]] = None) -> None:
        if not nodes and not edges:
            return
        await add_data_points(list(nodes), custom_edges=list(edges) if edges else None, ctx=ctx)
        stats["graph_nodes"] += len(nodes)
        stats["graph_edges"] += len(edges or [])
        logger.info("Streamed graph batch: %d nodes, %d edges", len(nodes), len(edges or []))

    # Pass 1: raw nodes stream straight to storage; entities buffer for merging.
    batch: List[Any] = []
    async for record in source.records():
        if record.kind == "raw_node":
            properties = record.properties or {}
            node = rehydrate_node(properties)
            if node.id in known_node_ids:
                continue
            known_node_ids.add(node.id)
            external_id = properties.get("id")
            if external_id:
                external_to_node_id[str(external_id)] = node.id
            batch.append(node)
            if len(batch) >= BATCH_NODE_TARGET:
                await flush(batch)
                batch = []
        elif record.kind == "entity":
            _register_entity(
                record,
                entity_types=entity_types,
                by_node_id=by_node_id,
                by_external_id=by_external_id,
                first_external_id=first_external_id,
            )
    await flush(batch)

    ordered_nodes = list(entity_types.values()) + list(by_node_id.values())
    for start in range(0, len(ordered_nodes), BATCH_NODE_TARGET):
        await flush(ordered_nodes[start : start + BATCH_NODE_TARGET])
    known_node_ids.update(node.id for node in ordered_nodes)
    for external_id, node in by_external_id.items():
        external_to_node_id[external_id] = node.id
    # Release the node objects; only the slim id registry survives into pass 2.
    entity_types, by_node_id, by_external_id, ordered_nodes = {}, {}, {}, []

    # Pass 2: facts resolve against the slim registry and flush as edge batches.
    stub_batch: List[Any] = []
    edge_batch: List[Tuple] = []
    stubbed: Set[UUID] = set()

    def resolve(ref: str) -> Tuple[Optional[UUID], Optional[Any]]:
        """Resolve a fact ref to a node id; returns (id, new_stub_entity_or_None)."""
        node_id = external_to_node_id.get(ref)
        if node_id is not None:
            return node_id, None
        candidate = generate_node_id(ref)
        if candidate in known_node_ids or candidate in stubbed:
            return candidate, None
        if _looks_like_uuid(ref):
            return None, None
        # Plain-name reference (cross-provider archives): create the entity.
        return candidate, Entity(id=candidate, name=ref, description=ref)

    async for record in source.records():
        if record.kind != "fact":
            continue
        subject_id, subject_stub = resolve(record.subject_ref)
        object_id, object_stub = resolve(record.object_ref)
        if subject_id is None or object_id is None:
            stats["skipped_facts"] += 1
            unresolved = [
                ref
                for ref, node_id in (
                    (record.subject_ref, subject_id),
                    (record.object_ref, object_id),
                )
                if node_id is None
            ]
            logger.warning(
                "Skipping fact %r (%s): unresolved UUID reference(s) %s",
                record.external_id,
                record.predicate,
                ", ".join(unresolved),
            )
            continue
        for stub in (subject_stub, object_stub):
            if stub is not None and stub.id not in stubbed:
                stub_batch.append(stub)
                stubbed.add(stub.id)
        edge_batch.append((subject_id, object_id, record.predicate, _fact_edge_properties(record)))
        if len(edge_batch) >= EDGE_BATCH_TARGET or len(stub_batch) >= BATCH_NODE_TARGET:
            await flush(stub_batch, edge_batch)
            stub_batch, edge_batch = [], []
    await flush(stub_batch, edge_batch)

    return stats


async def store_imported_graph(batches, ctx=None):
    """Pipeline task: persist translated graph batches via add_data_points."""
    from cognee.tasks.storage.add_data_points import add_data_points

    ctx = _provenance_ctx(ctx)
    if isinstance(batches, (dict, DataItem)):
        batches = [batches]

    for batch in batches:
        if isinstance(batch, DataItem):
            batch = batch.data
        await add_data_points(
            batch["nodes"],
            custom_edges=batch["edges"] or None,
            ctx=ctx,
        )
        logger.info(
            "Stored imported graph batch: %d nodes, %d edges",
            len(batch["nodes"]),
            len(batch["edges"]),
        )
    return batches
