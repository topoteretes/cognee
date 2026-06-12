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
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.migration.cogx import (
    COGXEntity,
    COGXEpisode,
    COGXFact,
    COGXRecord,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

logger = get_logger("migration.loader")

FACTS_PER_DIGEST = 200


@dataclass
class TranslationResult:
    data_items: List[DataItem] = field(default_factory=list)
    graph_batches: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
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


def _build_graph_batch(
    entities: List[COGXEntity], facts: List[COGXFact]
) -> Optional[Dict[str, Any]]:
    """Map entity/fact records onto native Entity DataPoints and edge tuples.

    Entity ids come from ``generate_node_id(name)`` — the same scheme cognify
    uses — so preserved facts merge into the existing graph vocabulary instead
    of forming a disconnected parallel graph.
    """
    if not entities and not facts:
        return None

    entity_types: Dict[str, EntityType] = {}
    by_external_id: Dict[str, Entity] = {}
    by_node_id: Dict[UUID, Entity] = {}

    def _entity_type_for(name: str) -> EntityType:
        key = name.lower()
        if key not in entity_types:
            entity_types[key] = EntityType(id=generate_node_id(name), name=name, description=name)
        return entity_types[key]

    def _register(entity: Entity, external_id: Optional[str]) -> Entity:
        existing = by_node_id.get(entity.id)
        if existing is None:
            by_node_id[entity.id] = entity
            existing = entity
        if external_id:
            by_external_id[external_id] = existing
        return existing

    for record in entities:
        description = record.description or record.name
        if record.aliases:
            description = f"{description} Also known as: {', '.join(record.aliases)}."
        entity = Entity(
            id=generate_node_id(record.name),
            name=record.name,
            description=description,
            is_a=_entity_type_for(record.entity_type) if record.entity_type else None,
        )
        _register(entity, record.external_id)

    def _resolve_ref(ref: str) -> Entity:
        entity = by_external_id.get(ref)
        if entity is not None:
            return entity
        # Unknown reference: treat it as an entity name.
        entity = Entity(id=generate_node_id(ref), name=ref, description=ref)
        return _register(entity, None)

    edges: List[Tuple[UUID, UUID, str, Dict[str, Any]]] = []
    for fact in facts:
        subject = _resolve_ref(fact.subject_ref)
        target = _resolve_ref(fact.object_ref)
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
        edges.append((subject.id, target.id, fact.predicate, properties))

    nodes: List[Any] = list(entity_types.values()) + list(by_node_id.values())
    return {"nodes": nodes, "edges": edges}


def translate_records(records: Iterable[COGXRecord], mode: str) -> TranslationResult:
    """Translate COGX records according to the import fidelity mode."""
    result = TranslationResult(cognify_data_items=mode != "preserve")
    entities: List[COGXEntity] = []
    facts: List[COGXFact] = []

    for record in records:
        result.counts[record.kind] = result.counts.get(record.kind, 0) + 1

        if record.kind == "document":
            result.data_items.append(_data_item_for(record, record.content, record.title))
        elif record.kind == "episode":
            result.data_items.append(_data_item_for(record, render_episode(record), record.title))
        elif record.kind == "memory":
            content = record.content
            if record.categories:
                content = f"{content}\nCategories: {', '.join(record.categories)}"
            result.data_items.append(_data_item_for(record, content))
        elif record.kind == "memory_block":
            content = f"{record.label}:\n{record.value}"
            result.data_items.append(_data_item_for(record, content, record.label))
        elif record.kind == "entity":
            entities.append(record)
        elif record.kind == "fact":
            facts.append(record)

    if mode == "re-derive":
        # Render the source's derived knowledge as digest documents so it is
        # not lost, and let cognify re-extract it.
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
        batch = _build_graph_batch(entities, facts)
        if batch:
            result.graph_batches.append(batch)

    return result


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
        extras=getattr(ctx, "extras", None),
    )


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
