"""Write the conflicts each review call found, and clear their pending-write marker last.

The step order in ``_write_conflict_changes`` is the safety argument of this file.
Every conflict a write touches is marked ``write_pending`` before anything
else is written, so a run that dies midway leaves a marker the next read finds
and repairs. The marker is only cleared at the end of the stream, once every
write has landed.
"""

import dataclasses
import json
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.engine.models import FactConflict
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.fact_conflicts import (
    CONFLICT_ABOUT,
    CONFLICT_CITES,
    CONFLICT_VALUE,
    STATUS_PRECEDENCE,
    read_conflict_marks,
)
from cognee.tasks.storage import add_data_points

from .accept_review import _collect_conflict_values
from .facts import get_endpoints
from .models import AcceptedConflict, ConflictsToFinalize, ConflictsToWrite, ReviewScope


@dataclasses.dataclass(frozen=True)
class ConflictChange:
    """One conflict this write rewrites or drops, with both sides resolved once."""

    conflict_id: str
    stored: dict  # the node as the graph holds it now; {} for one first seen here
    accepted: AcceptedConflict | None  # the reviewed replacement; None for a drop

    @property
    def dropped(self) -> bool:
        return self.accepted is None

    @property
    def fact_statuses(self) -> dict[str, str]:
        return self.accepted.fact_statuses if self.accepted else {}


def _preserve_user_resolutions(to_write: ConflictsToWrite) -> None:
    """An automatic review may refresh evidence, but cannot overturn a user's choice.

    Mutate the batch so its finalization receives the same protected nodes. Drops of
    deleted subjects still go through; LLM drops of a live user's decision do not.
    The choice covers only the facts the user weighed. Evidence reported after it is
    never marked rejected on the user's behalf: it follows each review exactly as
    under a conflict the review owns.
    """
    accepted = {str(item.conflict.id): item for item in to_write.accepted_conflicts}
    for conflict_id, stored in to_write.scope.conflicts.items():
        if stored.get("resolution_source") != "user":
            continue
        incoming = accepted.get(conflict_id)
        dropping = conflict_id in to_write.dropped_conflict_ids
        if incoming and incoming.conflict.resolution_source == "user":
            continue  # An explicit API resolution, including a revision of a prior choice.
        if not incoming and not dropping:
            continue
        if stored.get("selected_fact_id") not in to_write.scope.facts:
            continue  # Its evidence was forgotten; let the current review replace or drop it.
        if dropping and stored["about_id"] not in to_write.scope.entities:
            continue  # The subject itself was deleted.
        selected_fact_id = stored.get("selected_fact_id")
        decided = {selected_fact_id, *stored.get("rejected_fact_ids", [])}
        statuses: dict[str, str] = {}
        for fact_id, fact in to_write.scope.facts.items():
            held = next(
                (
                    mark.get("status")
                    for mark in read_conflict_marks(fact["properties"])
                    if str(mark.get("conflict_id")) == conflict_id
                ),
                None,
            )
            reported = incoming.fact_statuses.get(fact_id) if incoming else None
            if held is None and reported is None:
                continue
            if fact_id == selected_fact_id:
                statuses[fact_id] = "current"
            elif fact_id in decided:
                statuses[fact_id] = "superseded"
            elif reported or fact_id not in to_write.shown_fact_ids:
                # A shown fact the review no longer labels has left the conflict.
                statuses[fact_id] = reported or held
        rejected = sorted(fid for fid in statuses if fid in decided and fid != selected_fact_id)
        # Every list describes the facts that exist now. A forgotten fact leaves no
        # trace: a stale text would keep rewriting fact hits, a stale chunk sends the
        # conflict back to review on every run, a stale value keeps a deleted node in scope.
        replacement = _rebuild_stored_conflict(
            stored,
            rejected_fact_ids=rejected,
            rejected_fact_texts=sorted(
                {to_write.scope.facts[fid]["properties"].get("edge_text", "") for fid in rejected}
                - {""}
            ),
            sources=sorted(_collect_cited_sources(statuses, to_write.scope.facts)),
            values=sorted(_collect_conflict_values(stored["about_id"], statuses, to_write.scope)),
        )
        accepted[conflict_id] = AcceptedConflict(replacement, statuses)
        if dropping:
            to_write.dropped_conflict_ids.remove(conflict_id)
    to_write.accepted_conflicts[:] = list(accepted.values())


def _collect_conflict_changes(to_write: ConflictsToWrite) -> list[ConflictChange]:
    """Every conflict this write touches, stored and reviewed sides looked up once."""
    accepted_by_id = {str(item.conflict.id): item for item in to_write.accepted_conflicts}
    drops = set(to_write.dropped_conflict_ids)
    stored = to_write.scope.conflicts
    return [
        ConflictChange(conflict_id, stored.get(conflict_id) or {}, accepted_by_id.get(conflict_id))
        # Sorted so a fact's rewritten conflict_marks land in a reproducible order;
        # nothing reads that order, but an unsorted set made the written list, and
        # therefore whether the edge was written at all, depend on the hash seed.
        for conflict_id in sorted(set(accepted_by_id) | drops)
    ]


def _rebuild_stored_conflict(stored: dict, **overrides) -> FactConflict:
    """Rebuild a conflict from the properties a graph read returned.

    Neo4j hands ``metadata`` back as JSON text rather than a dict.
    """
    properties = dict(stored)
    if isinstance(properties.get("metadata"), str):
        properties["metadata"] = json.loads(properties["metadata"])
    return FactConflict(**{**properties, **overrides})


def _build_conflict_storage_context(ctx, dataset_id: str):
    """Give review-written nodes a stable owner of their own.

    ``add_data_points`` attributes a write to (dataset, data item), and memify's
    data item carries no id. Without this the conflict nodes would be written
    unattributed and a dataset delete would leave them behind. A uuid5 of the
    dataset is the same owner on every run.
    """
    data_id = uuid5(NAMESPACE_URL, f"cognee:review-conflicts:{dataset_id}")
    return dataclasses.replace(ctx, data_item=SimpleNamespace(id=data_id))


# --- the steps of one write, in the order they must run ----------------------


async def _mark_conflicts_pending(changes: list[ConflictChange], conflict_ctx) -> None:
    """Mark every touched conflict as a pending write before any dependent write starts."""
    nodes = []
    for change in changes:
        if change.stored:
            nodes.append(_rebuild_stored_conflict(change.stored, write_pending=True))
        elif not change.dropped:
            nodes.append(change.accepted.conflict.model_copy(update={"write_pending": True}))
    if nodes:
        await add_data_points(nodes, ctx=conflict_ctx)


def _restate_marks(
    marks: list[dict], fact_id: str, to_write: ConflictsToWrite, changes: list[ConflictChange]
) -> list[dict]:
    """This fact's marks once this write has restated every conflict it was shown for.

    A shown fact the review did not label has left that conflict, so its mark is
    removed with nothing put back.
    """
    for change in changes:
        status = change.fact_statuses.get(fact_id)
        retired_user_choice = (
            change.stored.get("resolution_source") == "user"
            and change.stored.get("selected_fact_id") not in to_write.scope.facts
        )
        restated = (
            status is not None
            or change.dropped
            or fact_id in to_write.shown_fact_ids
            or retired_user_choice
        )
        if not restated:
            continue
        marks = [mark for mark in marks if str(mark["conflict_id"]) != change.conflict_id]
        if status:
            mark = {"conflict_id": change.conflict_id, "status": status}
            conflict = change.accepted.conflict
            # Only the facts the user weighed carry the decision; evidence reported after
            # it keeps a plain review mark (see _preserve_user_resolutions).
            if conflict.resolution_source == "user" and fact_id in (
                conflict.selected_fact_id,
                *conflict.rejected_fact_ids,
            ):
                mark.update(
                    resolution_source="user",
                    dataset_id=conflict.dataset_id,
                    selected_fact=conflict.selected_fact,
                )
            marks.append(mark)
    return marks


def _build_changed_fact_properties(
    to_write: ConflictsToWrite, changes: list[ConflictChange]
) -> dict[str, dict]:
    """New property dicts for the facts whose marks or source date this write changes."""
    changed: dict[str, dict] = {}
    for fact_id, fact in to_write.scope.facts.items():
        properties = fact["properties"]
        marks = list(properties.get("conflict_marks") or [])
        restated = _restate_marks(marks, fact_id, to_write, changes)
        # A shown fact is also rewritten when its stored date is not the date read now,
        # because the review judged it with that date. Every rewrite stores that date.
        redated = fact_id in to_write.shown_fact_ids and (
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


async def _update_fact_properties(
    graph_engine, to_write: ConflictsToWrite, changes: list[ConflictChange]
) -> None:
    changed = _build_changed_fact_properties(to_write, changes)
    if not changed:
        return
    scope = to_write.scope
    await graph_engine.add_edges(
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


def _is_stored_text_replaced(change: ConflictChange) -> bool:
    """The stored conflict's explanation is not the one this write stores."""
    if not change.stored:
        return False
    if change.dropped:
        return True
    return EdgeType.id_for(change.stored["text"]) != EdgeType.id_for(change.accepted.conflict.text)


async def _delete_replaced_text_embeddings(vector_engine, changes: list[ConflictChange]) -> None:
    """Drop the embedding of every conflict explanation this write replaced.

    An explanation is stored as an EdgeType relationship name and indexed, so a
    stale one keeps matching searches after the conflict it explained was
    rewritten or dropped.
    """
    texts = {change.stored["text"] for change in changes if _is_stored_text_replaced(change)}
    if not texts:
        return
    try:
        await vector_engine.delete_data_points(
            "EdgeType_relationship_name", [str(EdgeType.id_for(text)) for text in sorted(texts)]
        )
    except CollectionNotFoundError:
        # A dataset whose edge texts were never indexed has no collection to clean.
        pass


async def _delete_value_and_citation_links(graph_engine, changes: list[ConflictChange]) -> None:
    """Clear a rewritten conflict's old value and citation links before writing new ones.

    A dropped conflict keeps its links until finalizing deletes its node with them.
    """
    conflict_ids = [change.conflict_id for change in changes if not change.dropped]
    if not conflict_ids:
        return
    # Read actual links: a previous write may have stopped between node and edges.
    _, edges = await graph_engine.get_neighborhood(conflict_ids, depth=1)
    triples = [
        EdgeIdentity(str(source), str(target), relationship)
        for source, target, relationship, *_ in edges
        if str(source) in conflict_ids and relationship in (CONFLICT_VALUE, CONFLICT_CITES)
    ]
    if not triples:
        return
    await graph_engine.delete_edge_triples(triples)


def _choose_value_statuses(accepted: AcceptedConflict, facts: dict[str, dict]) -> dict[str, str]:
    """One status per conflict value: the strongest any labelled fact gave it.

    The strongest is the first in ``STATUS_PRECEDENCE``, which also picks a fact's status.
    """
    statuses: dict[str, str] = {}
    for fact_id, status in accepted.fact_statuses.items():
        fact = facts[fact_id]
        for value_id in accepted.conflict.values:
            if value_id not in get_endpoints(fact):
                continue
            held = statuses.get(value_id)
            if held is None or STATUS_PRECEDENCE.index(status) < STATUS_PRECEDENCE.index(held):
                statuses[value_id] = status
    return statuses


def _collect_cited_sources(fact_ids, facts: dict[str, dict]) -> dict[str, dict]:
    """One source record per chunk the given facts cite now."""
    return {
        source["chunk_id"]: source for fact_id in fact_ids for source in facts[fact_id]["sources"]
    }


def _build_conflict_edges(accepted: AcceptedConflict, facts: dict[str, dict]) -> list[tuple]:
    """A conflict's subject link, its value links with their statuses, and its citations."""
    conflict = accepted.conflict
    conflict_id = str(conflict.id)
    text = {"edge_text": conflict.text}
    return [
        (conflict_id, str(conflict.about_id), CONFLICT_ABOUT, text),
        *(
            (conflict_id, value_id, CONFLICT_VALUE, {**text, "status": status})
            for value_id, status in sorted(_choose_value_statuses(accepted, facts).items())
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
            for chunk_id, source in sorted(
                _collect_cited_sources(accepted.fact_statuses, facts).items()
            )
        ),
    ]


async def _save_conflict_nodes(
    changes: list[ConflictChange], scope: ReviewScope, conflict_ctx
) -> None:
    """Write the reviewed conflict nodes with their subject, value and citation links."""
    nodes: list[FactConflict] = []
    edges: list[tuple] = []
    for change in changes:
        if change.dropped:
            continue
        nodes.append(change.accepted.conflict.model_copy(update={"write_pending": True}))
        edges.extend(_build_conflict_edges(change.accepted, scope.facts))
    if nodes:
        await add_data_points(nodes, custom_edges=edges, ctx=conflict_ctx)


def _update_scope_conflicts(changes: list[ConflictChange], scope: ReviewScope) -> None:
    """Show later calls each conflict as this write left it: rewritten, or gone."""
    for change in changes:
        if change.dropped:
            scope.conflicts.pop(change.conflict_id, None)
        else:
            scope.conflicts[change.conflict_id] = change.accepted.conflict.model_dump(mode="json")


# --- the enrichment task -----------------------------------------------------


async def _write_conflict_changes(unified, to_write: ConflictsToWrite, ctx) -> None:
    _preserve_user_resolutions(to_write)
    changes = _collect_conflict_changes(to_write)
    conflict_ctx = _build_conflict_storage_context(ctx, to_write.scope.dataset_id)

    # The pending-write marker goes first: if a step below fails, the next run finds the
    # marker and reviews the conflict again.
    await _mark_conflicts_pending(changes, conflict_ctx)
    await _update_fact_properties(unified.graph, to_write, changes)
    await _delete_replaced_text_embeddings(unified.vector, changes)
    await _delete_value_and_citation_links(unified.graph, changes)
    await _save_conflict_nodes(changes, to_write.scope, conflict_ctx)
    _update_scope_conflicts(changes, to_write.scope)


async def _finalize_run(graph_engine, to_finalize: ConflictsToFinalize) -> None:
    """Clear or delete every conflict the run touched; no further write depends on them."""
    for accepted in to_finalize.accepted_conflicts:
        conflict = accepted.conflict.model_copy(
            update={"write_pending": accepted.conflict.cache_cleanup_pending}
        )
        await graph_engine.add_nodes([conflict])
    for conflict_id in to_finalize.dropped_conflict_ids:
        await graph_engine.delete_nodes([conflict_id])


async def write_conflicts(items, ctx) -> None:
    """Write one review call's conflicts, or finalize the run on ``ConflictsToFinalize``."""
    item = items[0]  # batch_size=1: the enrichment stream hands over one item at a time.
    unified = await get_unified_engine()
    if isinstance(item, ConflictsToFinalize):
        await _finalize_run(unified.graph, item)
    else:
        await _write_conflict_changes(unified, item, ctx)
