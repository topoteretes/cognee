"""Load facts and every document that cites them, without changing the graph.

``read_entity_facts`` at the bottom of the file runs the reads in order: this
dataset's stored conflicts, the entities the run selected, one hop around them,
the facts on its edges, the types of the values those facts name, and the
documents that cite each fact. Which entities are worth a call is the reviewer's
decision.
"""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime, timezone
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

from .facts import CHUNK_STATEMENT, TIMESTAMP, get_endpoints, parse_date
from .models import ReviewScope

logger = get_logger("review_conflicts")

# Edges that say nothing a review can weigh: its own bookkeeping, the links a
# previous contradiction pass left behind, and type membership, which is read
# separately as an entity's types.
NON_FACT_RELATIONSHIPS = {"contradicts", "is_a"}


# --- reading the graph -----------------------------------------------------


def _to_property_dicts(nodes) -> list[dict]:
    """Map an adapter's (node_id, properties) rows to property dicts carrying a string id."""
    return [{**properties, "id": str(node_id)} for node_id, properties in nodes]


async def _read_nodes_or_raise(graph_engine, node_ids) -> list[dict]:
    """A failed read must not look like deleted entities or a completed empty review."""
    if not node_ids:
        return []
    # get_nodes() is the by-id read, but Ladybug returns [] and Neptune a partial list when it
    # fails, and a node missing here deletes every FactConflict about it. The filtered read
    # raises; its discarded edge query is the price, and batching re-runs both per batch.
    nodes, _ = await graph_engine.get_filtered_graph_data([{"id": list(node_ids)}])
    return _to_property_dicts(nodes)


async def read_dataset_node_ids(graph_engine, dataset_id: str) -> set[str] | None:
    """This dataset's node ids when several datasets share the graph database, else None.

    With access control on, every dataset has a database of its own, so no read needs scoping.
    """
    if backend_access_control_enabled():
        return None
    if await stores_provenance_in_graph(graph_engine):
        return set(await graph_engine.find_node_source_refs_by_dataset(dataset_id))
    return {str(node.slug) for node in await get_dataset_related_nodes(UUID(dataset_id))}


async def _read_neighborhood(
    graph_engine,
    node_ids: Collection[str],
    dataset_node_ids: set[str] | None,
    edge_types: list[str] | None = None,
) -> tuple[dict[str, dict], list[tuple]]:
    """One hop around ``node_ids``: the dataset's non-internal nodes it reaches, and its edges.

    The adapter returns every edge among the nodes it reached, not only the edges at
    ``node_ids``; ``edge_types`` limits which neighbours are reached. Empty ``node_ids``
    cost no adapter call.
    """
    if not node_ids:
        return {}, []
    nodes, edges = await graph_engine.get_neighborhood(
        sorted(node_ids), depth=1, edge_types=edge_types
    )
    kept = {}
    for node_id, properties in nodes:
        node_id = str(node_id)
        if not is_internal_node(properties) and (
            dataset_node_ids is None or node_id in dataset_node_ids
        ):
            kept[node_id] = {**properties, "id": node_id}
    return kept, list(edges)


def _is_reviewable_entity(node: dict) -> bool:
    return node.get("type") == "Entity" and not is_internal_node(node)


# --- the conflicts already stored -------------------------------------------


def _to_conflict_record(node_id, properties: dict) -> dict:
    """A stored FactConflict as plain property dict, with its list fields read back."""
    record = {**properties, "id": str(node_id)}
    for name in ("values", "sources", "rejected_fact_ids", "rejected_fact_texts"):
        record[name] = [str(value) for value in read_list_property(properties, name)]
        record.pop(f"{name}_json", None)
    return record


@dataclass(frozen=True)
class StoredConflicts:
    """Every stored conflict of one dataset, and which of them need repair."""

    by_id: dict[str, dict]
    ids_to_review: list[str]  # its write did not finish, or a chunk it cites is gone
    ids_to_drop: list[str]  # its subject is gone, no longer an entity, or outside the dataset

    def collect_subject_ids_to_review(self) -> set[str]:
        """The subject of every conflict to review, so the run selects it again."""
        return {str(self.by_id[conflict_id]["about_id"]) for conflict_id in self.ids_to_review}

    def select_touching(self, entity_ids: Collection[str]) -> dict[str, dict]:
        """Every conflict about or valued at one of these entities, and every one to repair.

        A conflict to drop touches no entity of the run, but the writer needs its stored node.
        """
        repair_ids = {*self.ids_to_review, *self.ids_to_drop}
        return {
            conflict_id: conflict
            for conflict_id, conflict in self.by_id.items()
            if conflict_id in repair_ids
            or str(conflict["about_id"]) in entity_ids
            or any(value in entity_ids for value in conflict["values"])
        }


async def read_stored_conflicts(
    graph_engine, dataset_id, dataset_node_ids: set[str] | None
) -> StoredConflicts:
    """Read this dataset's stored conflicts, and find those to review again and to drop.

    A conflict is reviewed again when its write did not finish or a chunk it cites is
    gone. It is dropped when its subject is gone, is no longer an entity, or is not
    among ``dataset_node_ids``: a run reads only this dataset's nodes, so it could never
    review that conflict again. Both are found even after citations were replaced.
    """
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["FactConflict"]}])
    conflicts = {
        str(node_id): _to_conflict_record(node_id, properties)
        for node_id, properties in nodes
        if str(properties.get("dataset_id")) == str(dataset_id)
    }
    required = {
        node_id
        for conflict in conflicts.values()
        for node_id in [str(conflict["about_id"]), *conflict["sources"]]
    }
    existing = {
        str(node["id"]): node for node in await _read_nodes_or_raise(graph_engine, sorted(required))
    }
    ids_to_review, ids_to_drop = [], []
    for conflict_id, conflict in conflicts.items():
        subject_id = str(conflict["about_id"])
        subject = existing.get(subject_id)
        if (
            not subject
            or not _is_reviewable_entity(subject)
            or (dataset_node_ids is not None and subject_id not in dataset_node_ids)
        ):
            ids_to_drop.append(conflict_id)
        elif conflict.get("write_pending") or any(
            source not in existing for source in conflict["sources"]
        ):
            ids_to_review.append(conflict_id)
    return StoredConflicts(conflicts, ids_to_review, ids_to_drop)


# --- choosing the entities and the values their facts name -----------------


async def _read_entities(
    graph_engine,
    entity_ids: Collection[str] | None,
    subject_ids_to_review: set[str],
    dataset_node_ids: set[str] | None,
) -> dict[str, dict]:
    """The entities this run reads, in the order the review calls them.

    Those asked for, or every entity when none are, and the subjects of the conflicts
    to review, limited to this dataset's existing entities.
    """
    by_id: dict[str, dict] = {}
    if entity_ids is None:
        entity_nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
        by_id = {node["id"]: node for node in _to_property_dicts(entity_nodes)}
        selected = set(by_id)
    else:
        selected = {str(node_id) for node_id in entity_ids}
    selected |= subject_ids_to_review
    if dataset_node_ids is not None:
        selected &= dataset_node_ids
    # Subjects to review and scoped ids are not in the typed read; deleted ones stay absent.
    for node in await _read_nodes_or_raise(graph_engine, sorted(selected - by_id.keys())):
        by_id[node["id"]] = node
    # Sorted because this order decides which subjects share one review call, and
    # adapter row order is not reproducible.
    return {
        node_id: by_id[node_id]
        for node_id in sorted(selected)
        if node_id in by_id and _is_reviewable_entity(by_id[node_id])
    }


def _collect_value_ids(
    facts: dict[str, dict], nodes: dict[str, dict], entity_ids: Collection[str]
) -> set[str]:
    """The other entities these facts name: values, whose types grouping and naming need."""
    return {
        endpoint
        for fact in facts.values()
        for endpoint in get_endpoints(fact)
        if endpoint not in entity_ids and nodes[endpoint].get("type") == "Entity"
    }


def _attach_entity_types(nodes: dict[str, dict], edges: list[tuple]) -> None:
    """Replace each entity's types with the names its ``is_a`` edges reach."""
    entities = {node_id: node for node_id, node in nodes.items() if node.get("type") == "Entity"}
    for entity in entities.values():
        entity["types"] = []
    for source, target, relationship, _ in edges:
        source, target = str(source), str(target)
        if relationship != "is_a" or source not in entities or target not in nodes:
            continue
        name = nodes[target].get("name")
        if name and name not in entities[source]["types"]:
            entities[source]["types"].append(name)


# --- turning edges into facts, with their citing documents and dates --------


def _select_fact_edges(nodes: dict[str, dict], edges: list[tuple]):
    """The edges a review can weigh: entity and date claims, and chunk statements with text."""
    for source, target, relationship, properties in edges:
        source, target = str(source), str(target)
        if source not in nodes or target not in nodes:
            continue
        if is_conflict_edge(relationship) or relationship in NON_FACT_RELATIONSHIPS:
            continue
        source_type = nodes[source].get("type")
        target_type = nodes[target].get("type")
        if (source_type == "Entity" and target_type in ("Entity", TIMESTAMP)) or (
            relationship == CHUNK_STATEMENT
            and source_type == "DocumentChunk"
            and target_type == "Entity"
            and properties.get("edge_text")
        ):
            yield source, target, relationship, properties


def _build_facts(
    nodes: dict[str, dict], edges: list[tuple], entity_ids: Collection[str]
) -> dict[str, dict]:
    """One fact per edge a review can weigh with one of ``entity_ids`` at an end.

    Keyed by the edge object id its evidence rows are written under. An edge between
    two neighbours is left out: no later step reads a fact that touches none of the
    run's entities.
    """
    facts: dict[str, dict] = {}
    for source, target, relationship, properties in _select_fact_edges(nodes, edges):
        if source not in entity_ids and target not in entity_ids:
            continue
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


def _normalize_date(value, warned: set[str]) -> str | None:
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


async def _attach_source_evidence(facts: dict[str, dict], dataset_id: str) -> None:
    """Give every fact its citing documents, its latest stated date and when it was last seen."""
    warned: set[str] = set()
    date_key = get_improve_config().effective_date_key
    for record in await get_edge_sources(UUID(dataset_id), [UUID(edge_id) for edge_id in facts]):
        fact = facts[str(record.edge_id)]
        effective_date = _normalize_date((record.external_metadata or {}).get(date_key), warned)
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
            or parse_date(effective_date) > parse_date(fact["effective_date"])
        ):
            fact["effective_date"] = effective_date
        if record.observed_at is not None:
            observed_at = parse_date(record.observed_at)
            fact["observed_at"] = max(fact["observed_at"] or observed_at, observed_at)


# --- the extraction task ----------------------------------------------------


def _get_dataset_id(ctx: PipelineContext | None) -> str:
    """The id of the dataset this pipeline runs on; a run without one is an error."""
    dataset = getattr(ctx, "dataset", None)
    if dataset is None:
        raise ValueError("Conflict review requires a dataset context")
    return str(getattr(dataset, "id", dataset))


async def read_entity_facts(
    data, entity_ids=None, since=None, ctx: PipelineContext | None = None
) -> ReviewScope:
    """Read one run's entities, their facts, and the conflicts stored about them."""
    dataset_id = _get_dataset_id(ctx)
    graph_engine = await get_graph_engine()
    dataset_node_ids = await read_dataset_node_ids(graph_engine, dataset_id)
    stored_conflicts = await read_stored_conflicts(graph_engine, dataset_id, dataset_node_ids)
    entities = await _read_entities(
        graph_engine,
        entity_ids,
        stored_conflicts.collect_subject_ids_to_review(),
        dataset_node_ids,
    )

    # One hop reads the entities' claims, dates, chunk statements and own types. Their
    # values need only their types, so the second read follows is_a edges alone.
    neighbors, edges = await _read_neighborhood(graph_engine, entities, dataset_node_ids)
    nodes = {**entities, **neighbors}
    facts = _build_facts(nodes, edges, entities)
    value_ids = _collect_value_ids(facts, nodes, entities)
    type_nodes, type_edges = await _read_neighborhood(
        graph_engine, value_ids, dataset_node_ids, edge_types=["is_a"]
    )
    nodes.update(type_nodes)
    _attach_entity_types(nodes, edges + type_edges)
    await _attach_source_evidence(facts, dataset_id)

    return ReviewScope(
        dataset_id,
        {entity_id: nodes[entity_id] for entity_id in entities},
        facts,
        stored_conflicts.select_touching(entities.keys() | value_ids),
        stored_conflicts.ids_to_drop,
        since,
        nodes,
    )
