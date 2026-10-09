"""Write the conflicts one reviewed batch found, and clear its retry marker last.

The step order in ``_write_reviewed_batch`` is the safety argument of this file.
Every conflict this batch touches is marked ``review_pending`` before anything
else is written, so a run that dies midway leaves a marker the next read finds
and repairs. The marker is only cleared at the end of the stream, and only for
conflicts whose subject and value descriptions all landed.
"""

import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.fact_conflicts import (
    CONFLICT_ABOUT,
    CONFLICT_CITES,
    CONFLICT_VALUE,
)
from cognee.tasks.storage import add_data_points

from .facts import endpoints
from .models import AcceptedConflict, ReviewBatch, ReviewScope

# A conflict value carries the strongest status any of its facts was given.
STATUS_PRIORITY = {"current": 0, "superseded": 1, "conflicting": 2}


@dataclasses.dataclass
class PendingConflict:
    """A written conflict whose retry marker clears once its entities are saved."""

    # None means this batch dropped the conflict: finalizing deletes the node.
    cleared_node: FactConflict | None
    blocked_by: set[str]


@dataclasses.dataclass
class ReviewWriteState:
    """Carried across the whole stream by the memify task, not per batch."""

    saved_entities: set[str] = dataclasses.field(default_factory=set)
    unreviewed_entity_ids: list[str] = dataclasses.field(default_factory=list)
    pending: dict[str, PendingConflict] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ConflictChange:
    """One conflict this batch rewrites or drops, with both sides resolved once."""

    conflict_id: str
    stored: dict  # the node as the graph holds it now; {} for one first seen here
    accepted: AcceptedConflict | None  # the reviewed replacement; None for a drop
    dropped: bool

    @property
    def fact_statuses(self) -> dict[str, str]:
        return self.accepted.fact_statuses if self.accepted else {}


def _changed_conflicts(batch: ReviewBatch) -> list[ConflictChange]:
    """Every conflict this batch touches, stored and reviewed sides looked up once."""
    accepted_by_id = {str(item.conflict.id): item for item in batch.conflicts}
    drops = set(batch.dropped_conflict_ids)
    stored = batch.scope.conflicts
    return [
        ConflictChange(
            conflict_id,
            stored.get(conflict_id) or {},
            accepted_by_id.get(conflict_id),
            conflict_id in drops,
        )
        # Sorted so a fact's rewritten conflict_marks land in a reproducible order;
        # nothing reads that order, but an unsorted set made the written list, and
        # therefore whether the edge was written at all, depend on the hash seed.
        for conflict_id in sorted(set(accepted_by_id) | drops)
    ]


def _datapoint_from_stored(model, stored: dict, **overrides):
    """Rebuild a DataPoint from the properties a graph read returned.

    Neo4j hands ``metadata`` back as JSON text rather than a dict, and a stored
    Entity's ``is_a`` and ``relations`` come back in shapes their field types
    reject — they are edges in the graph, re-derived by the writer on every save
    rather than round-tripped through the node.
    """
    properties = dict(stored)
    if isinstance(properties.get("metadata"), str):
        properties["metadata"] = json.loads(properties["metadata"])
    properties.pop("is_a", None)
    properties.pop("relations", None)
    return model(**{**properties, **overrides})


def _conflict_storage_context(ctx, dataset_id: str):
    """Give review-written nodes a stable owner of their own.

    ``add_data_points`` attributes a write to (dataset, data item), and memify's
    data item carries no id. Without this the conflict nodes would be written
    unattributed and a dataset delete would leave them behind. A uuid5 of the
    dataset is the same owner on every run. Entities keep the real ``ctx``, so
    they stay owned by the document they came from.
    """
    data_id = uuid5(NAMESPACE_URL, f"cognee:review-conflicts:{dataset_id}")
    return dataclasses.replace(ctx, data_item=SimpleNamespace(id=data_id))


# --- the steps of one batch, in the order they must run ----------------------


async def _mark_conflicts_pending(changes: list[ConflictChange], conflict_ctx) -> None:
    """Flag every touched conflict for repair before any dependent write starts."""
    nodes = []
    for change in changes:
        if change.stored:
            nodes.append(_datapoint_from_stored(FactConflict, change.stored, review_pending=True))
        elif change.accepted:
            nodes.append(change.accepted.conflict.model_copy(update={"review_pending": True}))
    if nodes:
        await add_data_points(nodes, ctx=conflict_ctx)


def _restated_marks(
    marks: list[dict], fact_id: str, batch: ReviewBatch, changes: list[ConflictChange]
) -> list[dict]:
    """This fact's marks once the batch has restated every conflict it was shown for.

    A shown fact the review did not label has left that conflict, so its mark is
    removed with nothing put back.
    """
    for change in changes:
        status = change.fact_statuses.get(fact_id)
        restated = status is not None or change.dropped or fact_id in batch.shown_fact_ids
        if not restated:
            continue
        marks = [mark for mark in marks if str(mark["conflict_id"]) != change.conflict_id]
        if status:
            marks.append({"conflict_id": change.conflict_id, "status": status})
    return marks


def _changed_fact_properties(
    scope: ReviewScope, batch: ReviewBatch, changes: list[ConflictChange]
) -> dict[str, dict]:
    """New property dicts for the facts whose marks or source date this batch changes."""
    changed: dict[str, dict] = {}
    for fact_id, fact in scope.facts.items():
        properties = fact["properties"]
        marks = list(properties.get("conflict_marks") or [])
        restated = _restated_marks(marks, fact_id, batch, changes)
        # Only covered/shown facts are changed; a trimmed fact keeps its old date too.
        redated = fact_id in batch.shown_fact_ids and (
            properties.get("effective_date") != fact["effective_date"]
        )
        if restated == marks and not redated:
            continue
        updated = {
            **properties,
            "conflict_marks": restated,
            "effective_date": fact["effective_date"],
        }
        updated.pop("conflict_marks_json", None)
        changed[fact_id] = updated
    return changed


async def _update_fact_marks(
    graph, scope: ReviewScope, batch: ReviewBatch, changes: list[ConflictChange]
) -> None:
    changed = _changed_fact_properties(scope, batch, changes)
    if not changed:
        return
    await graph.add_edges(
        [
            (
                scope.facts[fact_id]["source"],
                scope.facts[fact_id]["target"],
                scope.facts[fact_id]["relationship"],
                properties,
            )
            for fact_id, properties in changed.items()
        ]
    )
    # Only after the write lands: the next review call reads these marks from
    # this same scope object.
    for fact_id, properties in changed.items():
        scope.facts[fact_id]["properties"] = properties


async def _save_reviewed_entities(
    batch: ReviewBatch, scope: ReviewScope, ctx, state: ReviewWriteState
) -> None:
    """Rewrite each reviewed subject's description and stamp it as reviewed."""
    entities = [
        _datapoint_from_stored(
            Entity,
            scope.entities[entity_id],
            id=entity_id,
            description=description,
            conflicts_reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
        for entity_id, description in batch.descriptions.items()
    ]
    if entities:
        await add_data_points(entities, ctx=ctx)
        state.saved_entities.update(batch.descriptions)


def _replaces_stored_text(change: ConflictChange) -> bool:
    """The stored conflict's explanation is not the one this batch writes."""
    if not change.stored:
        return False
    if change.accepted is None:
        return True
    return EdgeType.id_for(change.stored["text"]) != EdgeType.id_for(change.accepted.conflict.text)


async def _delete_replaced_conflict_texts(vector, changes: list[ConflictChange]) -> None:
    """Drop the embedding of every conflict explanation this batch replaced.

    An explanation is stored as an EdgeType relationship name and indexed, so a
    stale one keeps matching searches after the conflict it explained was
    rewritten or dropped.
    """
    texts = {change.stored["text"] for change in changes if _replaces_stored_text(change)}
    if not texts:
        return
    try:
        await vector.delete_data_points(
            "EdgeType_relationship_name", [str(EdgeType.id_for(text)) for text in sorted(texts)]
        )
    except CollectionNotFoundError:
        # A dataset whose edge texts were never indexed has no collection to clean.
        pass


async def _remove_value_and_source_links(graph, conflict_ids: list[str]) -> None:
    """Clear a rewritten conflict's old value and citation links before writing new ones."""
    if not conflict_ids:
        return
    # Read actual links: a previous write may have stopped between node and edges.
    _, edges = await graph.get_neighborhood(conflict_ids, depth=1)
    triples = [
        EdgeIdentity(str(source), str(target), relationship)
        for source, target, relationship, *_ in edges
        if str(source) in conflict_ids and relationship in (CONFLICT_VALUE, CONFLICT_CITES)
    ]
    if not triples:
        return
    await graph.delete_edge_triples(triples)


def _value_statuses(accepted: AcceptedConflict, facts: dict[str, dict]) -> dict[str, str]:
    """One status per conflict value: the strongest any labelled fact gave it."""
    statuses: dict[str, str] = {}
    for fact_id, status in accepted.fact_statuses.items():
        fact = facts[fact_id]
        for value_id in accepted.conflict.values:
            if value_id not in endpoints(fact):
                continue
            held = statuses.get(value_id)
            if held is None or STATUS_PRIORITY[status] > STATUS_PRIORITY[held]:
                statuses[value_id] = status
    return statuses


def _cited_sources(accepted: AcceptedConflict, facts: dict[str, dict]) -> dict[str, dict]:
    """One source record per cited chunk, across every labelled fact."""
    return {
        source["chunk_id"]: source
        for fact_id in accepted.fact_statuses
        for source in facts[fact_id]["sources"]
    }


def _conflict_edges(accepted: AcceptedConflict, facts: dict[str, dict]) -> list[tuple]:
    """A conflict's subject link, its value links with their statuses, and its citations."""
    conflict = accepted.conflict
    conflict_id = str(conflict.id)
    text = {"edge_text": conflict.text}
    return [
        (conflict_id, str(conflict.about_id), CONFLICT_ABOUT, text),
        *(
            (conflict_id, value_id, CONFLICT_VALUE, {**text, "status": status})
            for value_id, status in sorted(_value_statuses(accepted, facts).items())
        ),
        *(
            (
                conflict_id,
                chunk_id,
                CONFLICT_CITES,
                {
                    **text,
                    "document": source["document"],
                    "effective_date": source["effective_date"],
                },
            )
            for chunk_id, source in sorted(_cited_sources(accepted, facts).items())
        ),
    ]


async def _save_conflict_nodes(
    changes: list[ConflictChange], scope: ReviewScope, conflict_ctx
) -> dict[str, FactConflict]:
    """Write the reviewed conflict nodes with their subject, value and citation links."""
    written: dict[str, FactConflict] = {}
    edges: list[tuple] = []
    for change in changes:
        if not change.accepted:
            continue
        conflict = change.accepted.conflict
        # Retain removed values until their descriptions have also been saved.
        values = sorted(set(conflict.values) | set(change.stored.get("values", [])))
        written[change.conflict_id] = conflict.model_copy(
            update={"review_pending": True, "values": values}
        )
        edges.extend(_conflict_edges(change.accepted, scope.facts))
    if written:
        await add_data_points(list(written.values()), custom_edges=edges, ctx=conflict_ctx)
    return written


def _blocking_entity_ids(change: ConflictChange, live_node_ids: set[str]) -> set[str]:
    """Entities whose rewritten description this conflict's retry marker waits on.

    A live value can be context-only on this run (e.g. synonym relationships).
    Keep its retry marker until a later read selects and rewrites that value.
    """
    ids = {str(change.stored.get("about_id", "")), *map(str, change.stored.get("values", []))}
    if change.accepted:
        conflict = change.accepted.conflict
        ids.update([str(conflict.about_id), *map(str, conflict.values)])
    return ids & live_node_ids


def _record_pending_conflicts(
    changes: list[ConflictChange],
    scope: ReviewScope,
    written: dict[str, FactConflict],
    state: ReviewWriteState,
) -> None:
    """Remember what each conflict still waits for, and what the graph now holds."""
    live_node_ids = scope.entities.keys() | scope.nodes.keys()
    for change in changes:
        cleared_node = change.accepted.conflict if change.accepted else None
        state.pending[change.conflict_id] = PendingConflict(
            cleared_node, _blocking_entity_ids(change, live_node_ids)
        )
        if cleared_node is not None:
            scope.conflicts[change.conflict_id] = written[change.conflict_id].model_dump(
                mode="json"
            )


# --- the enrichment task -----------------------------------------------------


async def _write_reviewed_batch(unified, batch: ReviewBatch, ctx, state: ReviewWriteState) -> None:
    scope = batch.scope
    changes = _changed_conflicts(batch)
    conflict_ctx = _conflict_storage_context(ctx, scope.dataset_id)
    rewritten_ids = [change.conflict_id for change in changes if change.accepted]

    # The marker goes first: every step below it may fail and be retried.
    await _mark_conflicts_pending(changes, conflict_ctx)
    await _update_fact_marks(unified.graph, scope, batch, changes)
    await _save_reviewed_entities(batch, scope, ctx, state)
    await _delete_replaced_conflict_texts(unified.vector, changes)
    await _remove_value_and_source_links(unified.graph, rewritten_ids)
    written = await _save_conflict_nodes(changes, scope, conflict_ctx)
    _record_pending_conflicts(changes, scope, written, state)


async def _finalize_run(graph, batch: ReviewBatch, state: ReviewWriteState) -> None:
    """Clear or delete every conflict whose dependent writes all landed."""
    state.unreviewed_entity_ids = batch.unreviewed_entity_ids
    # No further writes depend on a cleared marker or a deleted conflict.
    for conflict_id, pending in state.pending.items():
        if not pending.blocked_by <= state.saved_entities:
            continue
        if pending.cleared_node is None:
            await graph.delete_nodes([conflict_id])
        else:
            await graph.add_nodes(
                [pending.cleared_node.model_copy(update={"review_pending": False})]
            )


async def write_review_batch(batches, ctx, state: ReviewWriteState) -> None:
    """Write one review batch, or finalize the run when the stream's last batch arrives."""
    batch = batches[0]  # batch_size=1: the enrichment stream hands over one ReviewBatch.
    unified = await get_unified_engine()
    if batch.final:
        await _finalize_run(unified.graph, batch, state)
    else:
        await _write_reviewed_batch(unified, batch, ctx, state)
