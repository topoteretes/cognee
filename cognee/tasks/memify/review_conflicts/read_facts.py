"""Load complete, source-backed facts without changing the graph.

The stages below run in the order they are defined and are chained by
``read_entity_facts`` at the bottom of the file: resolve what this run may read,
pick its subjects, widen the view to the values they dispute, then turn the
edges that survived into dated, source-backed facts.
"""

from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal, NamedTuple
from uuid import UUID

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.engine.models.InternalDataPoint import is_internal_node
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.methods.get_dataset_related_nodes import get_dataset_related_nodes
from cognee.modules.graph.utils.fact_conflicts import (
    is_conflict_edge,
    read_conflict_marks,
    read_list_property,
)
from cognee.modules.improve.config import get_improve_config
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.provenance.edge_evidence.lookup import get_edge_sources
from cognee.shared.logging_utils import get_logger

from .facts import CHUNK_STATEMENT
from .models import ReviewScope

logger = get_logger("review_conflicts")

# Edges that say nothing a review can weigh: its own bookkeeping, the links a
# previous contradiction pass left behind, and type membership, which is read
# separately as a subject's types.
NON_FACT_RELATIONSHIPS = {"contradicts", "is_a"}


# --- reading the graph -----------------------------------------------------


def _as_property_dicts(nodes) -> list[dict]:
    """Map an adapter's (node_id, properties) rows to property dicts carrying a string id."""
    return [{**properties, "id": str(node_id)} for node_id, properties in nodes]


async def _get_nodes_or_raise(graph_engine, node_ids) -> list[dict]:
    """A failed read must not look like missing subjects or a completed empty review."""
    if not node_ids:
        return []
    # get_nodes() is the by-id read, but Ladybug returns [] and Neptune a partial list when it
    # fails, and a node missing here deletes every FactConflict about it. The filtered read
    # raises; its discarded edge query is the price, and batching re-runs both per batch.
    nodes, _ = await graph_engine.get_filtered_graph_data([{"id": list(node_ids)}])
    return _as_property_dicts(nodes)


async def _owned_node_ids(graph, dataset_id: str) -> set[str] | None:
    """The nodes this dataset owns, or None when the database is already its own."""
    if backend_access_control_enabled():
        return None
    if await stores_provenance_in_graph(graph):
        return set(await graph.find_node_source_refs_by_dataset(dataset_id))
    return {str(node.slug) for node in await get_dataset_related_nodes(UUID(dataset_id))}


@dataclass
class _OwnedSubgraph:
    """The slice of graph this review may read.

    Holds only non-internal nodes this dataset owns, and every edge the adapter
    returned around them. Two expansions widen the same view, so an edge can
    arrive twice; every reader keys or sets its results, so repeats change nothing.
    """

    owned_ids: set[str] | None
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[tuple] = field(default_factory=list)

    async def expand(self, graph, node_ids: Collection[str]) -> None:
        """Add one hop around ``node_ids``; an empty set costs no adapter call."""
        if not node_ids:
            return
        nodes, edges = await graph.get_neighborhood(sorted(node_ids), depth=1)
        for node_id, properties in nodes:
            node_id = str(node_id)
            if not is_internal_node(properties) and self._is_owned(node_id):
                self.nodes[node_id] = {**properties, "id": node_id}
        self.edges.extend(edges)

    def _is_owned(self, node_id: str) -> bool:
        """A per-dataset database needs no scope; a shared one is scoped to its own nodes."""
        return self.owned_ids is None or node_id in self.owned_ids


# --- the conflicts already stored -------------------------------------------


def _conflict_record(node_id, properties: dict) -> dict:
    """A stored FactConflict as plain property dict, with its list fields read back."""
    record = {**properties, "id": str(node_id)}
    for name in ("values", "sources"):
        record[name] = [str(value) for value in read_list_property(properties, name)]
        record.pop(f"{name}_json", None)
    return record


async def find_conflicts_with_lost_citations(
    graph_engine, dataset_id
) -> tuple[list[dict], list[dict]]:
    """Find deletion repairs and unfinished writes, even after citations were replaced."""
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["FactConflict"]}])
    conflicts = [
        _conflict_record(node_id, properties)
        for node_id, properties in nodes
        if str(properties.get("dataset_id")) == str(dataset_id)
    ]
    required = {
        node_id
        for conflict in conflicts
        for node_id in [str(conflict["about_id"]), *conflict["sources"]]
    }
    existing = (
        {
            str(node["id"]): node
            for node in await _get_nodes_or_raise(graph_engine, sorted(required))
        }
        if required
        else {}
    )
    surviving, drops = [], []
    for conflict in conflicts:
        subject = existing.get(str(conflict["about_id"]))
        if not subject or subject.get("type") != "Entity" or is_internal_node(subject):
            drops.append(conflict)
        elif conflict.get("review_pending") or any(
            source not in existing for source in conflict["sources"]
        ):
            surviving.append(conflict)
    return surviving, drops


def _repair_node_ids(repairs: dict[str, dict]) -> set[str]:
    """Every subject and value a repair names, so this run selects them again."""
    return {
        str(node_id)
        for conflict in repairs.values()
        for node_id in [conflict["about_id"], *conflict["values"]]
    }


def _conflicts_for_dataset(
    subgraph: _OwnedSubgraph, dataset_id: str, repairs: dict[str, dict]
) -> dict[str, dict]:
    """The repairs, plus every stored conflict of this dataset the read turned up."""
    conflicts = dict(repairs)
    for node_id, properties in subgraph.nodes.items():
        if (
            properties.get("type") == "FactConflict"
            and str(properties.get("dataset_id")) == dataset_id
        ):
            conflicts[node_id] = _conflict_record(node_id, properties)
    return conflicts


# --- choosing the subjects and the values they dispute ----------------------


def _is_reviewable_entity(node: dict) -> bool:
    return node.get("type") == "Entity" and not is_internal_node(node)


async def _select_subjects(
    graph, entity_ids, repair_ids: set[str], owned_ids: set[str] | None
) -> dict[str, dict]:
    """The entities this run reviews, in the order the review calls them."""
    by_id: dict[str, dict] = {}
    if entity_ids is None:
        entity_nodes, _ = await graph.get_filtered_graph_data([{"type": ["Entity"]}])
        by_id = {node["id"]: node for node in _as_property_dicts(entity_nodes)}
        selected = set(by_id)
    else:
        selected = {str(node_id) for node_id in entity_ids}
    selected |= repair_ids
    if owned_ids is not None:
        selected &= owned_ids
    # Repair subjects and scoped ids are not in the typed read; deleted ones stay absent.
    for node in await _get_nodes_or_raise(graph, sorted(selected - by_id.keys())):
        by_id[node["id"]] = node
    # Sorted because this order decides which subjects share one review call, and
    # adapter row order is not reproducible.
    return {
        node_id: by_id[node_id]
        for node_id in sorted(selected)
        if node_id in by_id and _is_reviewable_entity(by_id[node_id])
    }


class _ValueSlot(NamedTuple):
    """One side of one relationship on one entity: the place where values compete."""

    relationship: str
    side: Literal["source", "target"]
    entity_id: str


def _competing_value_ids(subgraph: _OwnedSubgraph) -> set[str]:
    """Ids that share a relationship slot with another id, i.e. the disputed values."""
    values_per_slot: dict[_ValueSlot, set[str]] = defaultdict(set)
    for source, target, relationship, _ in _fact_edges(subgraph):
        # A chunk statement names no competitor; only entity-to-entity claims do.
        if subgraph.nodes[source].get("type") != "Entity":
            continue
        values_per_slot[_ValueSlot(relationship, "source", source)].add(target)
        values_per_slot[_ValueSlot(relationship, "target", target)].add(source)
    return {
        value_id for values in values_per_slot.values() if len(values) > 1 for value_id in values
    }


def _entities_in_review_order(
    subgraph: _OwnedSubgraph, subjects: dict[str, dict], competing_ids: set[str]
) -> dict[str, dict]:
    """Subjects first, then the values they dispute, as the subgraph now holds them."""
    return {
        node_id: subgraph.nodes[node_id]
        for node_id in [*subjects, *sorted(competing_ids)]
        if node_id in subgraph.nodes
    }


def _attach_entity_types(entities: dict[str, dict], subgraph: _OwnedSubgraph) -> None:
    """Replace each entity's types with the names its ``is_a`` edges reach."""
    for entity in entities.values():
        entity["types"] = []
    for source, target, relationship, _ in subgraph.edges:
        source, target = str(source), str(target)
        if relationship != "is_a" or source not in entities or target not in subgraph.nodes:
            continue
        name = subgraph.nodes[target].get("name")
        if name and name not in entities[source]["types"]:
            entities[source]["types"].append(name)


# --- turning edges into dated, source-backed facts --------------------------


def _fact_edges(subgraph: _OwnedSubgraph):
    """The edges a review can weigh: entity claims, and chunk statements that carry text."""
    for source, target, relationship, properties in subgraph.edges:
        source, target = str(source), str(target)
        if source not in subgraph.nodes or target not in subgraph.nodes:
            continue
        if is_conflict_edge(relationship) or relationship in NON_FACT_RELATIONSHIPS:
            continue
        source_type = subgraph.nodes[source].get("type")
        target_type = subgraph.nodes[target].get("type")
        if source_type == target_type == "Entity" or (
            relationship == CHUNK_STATEMENT
            and source_type == "DocumentChunk"
            and target_type == "Entity"
            and properties.get("edge_text")
        ):
            yield source, target, relationship, properties


def _facts_from_edges(subgraph: _OwnedSubgraph) -> dict[str, dict]:
    """One fact per edge, keyed by the edge object id its evidence rows are written under."""
    facts: dict[str, dict] = {}
    for source, target, relationship, properties in _fact_edges(subgraph):
        properties = dict(properties)
        properties["conflict_marks"] = read_conflict_marks(properties)
        properties.pop("conflict_marks_json", None)
        edge_id = str(
            properties.get("edge_object_id")
            or generate_edge_object_id(source, target, relationship)
        )
        facts[edge_id] = {
            "id": edge_id,
            "source": source,
            "target": target,
            "relationship": relationship,
            "properties": properties,
            "sources": [],
            "effective_date": None,
            "observed_at": None,
        }
    return facts


def _normalized_date(value, warned: set[str]) -> str | None:
    """A document's stated date as ISO text in UTC, or None with one warning per value."""
    if value is None:
        return None
    try:
        if not isinstance(value, str):
            raise ValueError("Expected ISO date text")
        if len(value) == 10:
            return date.fromisoformat(value).isoformat()
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (
            timestamp.replace(tzinfo=timestamp.tzinfo or timezone.utc)
            .astimezone(timezone.utc)
            .isoformat()
        )
    except (TypeError, ValueError):
        key = repr(value)
        if key not in warned:
            warned.add(key)
            logger.warning("Ignoring invalid effective date: %s", value)
        return None


def _sortable_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


async def _attach_source_evidence(facts: dict[str, dict], dataset_id: str) -> None:
    """Give every fact its citing documents, its latest stated date and its latest write."""
    warned: set[str] = set()
    date_key = get_improve_config().effective_date_key
    for record in await get_edge_sources(UUID(dataset_id), [UUID(edge_id) for edge_id in facts]):
        fact = facts[str(record.edge_id)]
        effective_date = _normalized_date((record.external_metadata or {}).get(date_key), warned)
        fact["sources"].append(
            {
                "chunk_id": str(record.chunk_id),
                "data_id": str(record.data_id),
                "document": record.document_name,
                "effective_date": effective_date,
            }
        )
        if effective_date and (
            fact["effective_date"] is None
            or _sortable_date(effective_date) > _sortable_date(fact["effective_date"])
        ):
            fact["effective_date"] = effective_date
        if record.observed_at is not None:
            observed_at = record.observed_at.replace(
                tzinfo=record.observed_at.tzinfo or timezone.utc
            )
            fact["observed_at"] = max(fact["observed_at"] or observed_at, observed_at)


# --- the extraction task ----------------------------------------------------


def _require_dataset_id(ctx: PipelineContext | None) -> str:
    dataset = getattr(ctx, "dataset", None)
    if dataset is None:
        raise ValueError("Fact review requires a dataset context")
    return str(getattr(dataset, "id", dataset))


async def read_entity_facts(
    data, entity_ids=None, since=None, ctx: PipelineContext | None = None
) -> ReviewScope:
    """Collect one run's subjects, their source-backed facts, and the conflicts stored for them."""
    dataset_id = _require_dataset_id(ctx)
    graph = await get_graph_engine()
    owned_ids = await _owned_node_ids(graph, dataset_id)

    pending, dropped = await find_conflicts_with_lost_citations(graph, dataset_id)
    repairs = {conflict["id"]: conflict for conflict in [*pending, *dropped]}
    subjects = await _select_subjects(graph, entity_ids, _repair_node_ids(repairs), owned_ids)

    subgraph = _OwnedSubgraph(owned_ids, nodes=dict(subjects))
    await subgraph.expand(graph, subjects)
    # A value only competes once its own facts are loaded, so take a second hop.
    competing_ids = _competing_value_ids(subgraph) - subjects.keys()
    await subgraph.expand(graph, competing_ids)

    entities = _entities_in_review_order(subgraph, subjects, competing_ids)
    _attach_entity_types(entities, subgraph)
    facts = _facts_from_edges(subgraph)
    await _attach_source_evidence(facts, dataset_id)

    return ReviewScope(
        dataset_id,
        entities,
        facts,
        _conflicts_for_dataset(subgraph, dataset_id, repairs),
        [conflict["id"] for conflict in dropped],
        since,
        subgraph.nodes,
    )
