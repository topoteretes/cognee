from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.global_context_index.bucketing.graph.scoring import (
    compute_idf_from_counts,
    weighted_jaccard,
)
from cognee.tasks.storage import index_graph_edges

logger = get_logger("cross_connect_entities")

ENTITY_NAME_COLLECTION = "Entity_name"

# Inferred edges start with a low feedback weight so they can be told apart from
# extraction-time relationships and pruned first if they turn out to be noise.
INFERRED_EDGE_FEEDBACK_WEIGHT = 0.2

RELATION_INFERENCE_SYSTEM_PROMPT = (
    "You are linking entities in a knowledge graph. You are given two entities that are "
    "not yet connected but look related. Decide whether a real, direct relationship holds "
    "between them. If it does, name it as a short snake_case predicate reading source -> "
    "target (e.g. works_for, located_in, part_of). If the entities are merely similar or "
    "share context without a concrete relationship, mark them unrelated."
)


class InferredRelation(BaseModel):
    related: bool = Field(description="Whether a real, direct relationship holds between the pair")
    relationship_name: str = Field(description="snake_case predicate reading source -> target")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that the relationship holds")


async def get_entity_nodes(data: Any) -> list[tuple[str, dict[str, Any]]]:
    graph_engine = await get_graph_engine()
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    return [(str(node_id), props) for node_id, props in nodes]


async def cross_connect_entities(
    entities: list[tuple[str, dict[str, Any]]],
    similarity_threshold: float = 0.5,
    overlap_threshold: float = 0.2,
    max_new_edges_per_node: int = 5,
    confidence_threshold: float = 0.7,
    dry_run: bool = False,
) -> dict[str, Any]:
    if len(entities) < 2:
        return {"proposed": [], "written": 0, "dry_run": dry_run}

    graph_engine = await get_graph_engine()
    props_by_id = {entity_id: props for entity_id, props in entities}

    neighbors_by_entity = await _load_neighbor_sets(graph_engine, list(props_by_id))
    idf_weights = _neighbor_idf(neighbors_by_entity)

    candidate_pairs = _overlap_candidate_pairs(neighbors_by_entity, idf_weights, overlap_threshold)
    candidate_pairs |= await _vector_candidate_pairs(
        props_by_id, similarity_threshold, max_new_edges_per_node + 1
    )

    unlinked_pairs = [
        pair for pair in sorted(candidate_pairs) if not _already_linked(pair, neighbors_by_entity)
    ]

    edges = []
    proposed = []
    new_edges_per_node: dict[str, int] = defaultdict(int)
    for source_id, target_id in unlinked_pairs:
        if (
            new_edges_per_node[source_id] >= max_new_edges_per_node
            or new_edges_per_node[target_id] >= max_new_edges_per_node
        ):
            continue

        relation = await _infer_relation(props_by_id[source_id], props_by_id[target_id])
        if not relation.related or relation.confidence < confidence_threshold:
            continue

        edges.append(_build_edge(source_id, target_id, relation))
        proposed.append(
            {
                "source": source_id,
                "target": target_id,
                "relationship_name": relation.relationship_name,
                "confidence": relation.confidence,
            }
        )
        new_edges_per_node[source_id] += 1
        new_edges_per_node[target_id] += 1

    if edges and not dry_run:
        edges = await _drop_existing_edges(graph_engine, edges)
        await graph_engine.add_edges(edges)
        await index_graph_edges(edges)

    logger.info(
        "Cross-connect proposed %d edge(s)%s",
        len(proposed),
        " (dry run)" if dry_run else "",
    )
    return {"proposed": proposed, "written": 0 if dry_run else len(edges), "dry_run": dry_run}


async def _load_neighbor_sets(graph_engine, entity_ids: list[str]) -> dict[str, set[str]]:
    neighbor_sets: dict[str, set[str]] = {}
    for entity_id in entity_ids:
        neighbors = await graph_engine.get_neighbors(entity_id)
        neighbor_sets[entity_id] = {str(n["id"]) for n in neighbors if n.get("id")}
    return neighbor_sets


def _neighbor_idf(neighbors_by_entity: dict[str, set[str]]) -> dict[str, float]:
    neighbor_counts: Counter[str] = Counter()
    for neighbors in neighbors_by_entity.values():
        neighbor_counts.update(neighbors)
    return compute_idf_from_counts(len(neighbors_by_entity), neighbor_counts)


def _overlap_candidate_pairs(
    neighbors_by_entity: dict[str, set[str]],
    idf_weights: dict[str, float],
    overlap_threshold: float,
) -> set[tuple[str, str]]:
    # Only pivot on neighbors that carry weight; ubiquitous ones (e.g. shared
    # EntityType) have idf 0 and would otherwise pair up half the graph.
    neighbor_to_entities: dict[str, set[str]] = defaultdict(set)
    for entity_id, neighbors in neighbors_by_entity.items():
        for neighbor_id in neighbors:
            if idf_weights.get(neighbor_id, 0.0) > 0:
                neighbor_to_entities[neighbor_id].add(entity_id)

    pairs = set()
    for shared_entities in neighbor_to_entities.values():
        for left, right in combinations(sorted(shared_entities), 2):
            if (left, right) in pairs:
                continue
            overlap = weighted_jaccard(
                neighbors_by_entity[left], neighbors_by_entity[right], idf_weights
            )
            if overlap >= overlap_threshold:
                pairs.add((left, right))
    return pairs


async def _vector_candidate_pairs(
    props_by_id: dict[str, dict[str, Any]],
    similarity_threshold: float,
    limit: int,
) -> set[tuple[str, str]]:
    vector_engine = get_vector_engine()
    if not await vector_engine.has_collection(ENTITY_NAME_COLLECTION):
        return set()

    pairs = set()
    for entity_id, props in props_by_id.items():
        name = props.get("name")
        if not name:
            continue

        results = await vector_engine.search(ENTITY_NAME_COLLECTION, query_text=name, limit=limit)
        for result in results:
            other_id = str(result.id)
            if other_id == entity_id or other_id not in props_by_id:
                continue
            # score is a cosine distance (lower is closer); flip it to a similarity.
            if 1.0 - float(result.score) < similarity_threshold:
                continue
            pairs.add(tuple(sorted((entity_id, other_id))))
    return pairs


async def _drop_existing_edges(graph_engine, edges):
    guarded = []
    for edge in edges:
        source_id, target_id, relationship_name, _ = edge
        if not await graph_engine.has_edge(source_id, target_id, relationship_name):
            guarded.append(edge)
    return guarded


def _already_linked(pair: tuple[str, str], neighbors_by_entity: dict[str, set[str]]) -> bool:
    left, right = pair
    return right in neighbors_by_entity.get(left, set()) or left in neighbors_by_entity.get(
        right, set()
    )


async def _infer_relation(
    source_props: dict[str, Any], target_props: dict[str, Any]
) -> InferredRelation:
    try:
        return await LLMGateway.acreate_structured_output(
            text_input=_describe_pair(source_props, target_props),
            system_prompt=RELATION_INFERENCE_SYSTEM_PROMPT,
            response_model=InferredRelation,
        )
    except Exception as error:
        logger.warning("Relation inference failed: %s", error)
        return InferredRelation(related=False, relationship_name="", confidence=0.0)


def _describe_pair(source_props: dict[str, Any], target_props: dict[str, Any]) -> str:
    return (
        f"Source entity: {_describe_entity(source_props)}\n"
        f"Target entity: {_describe_entity(target_props)}"
    )


def _describe_entity(props: dict[str, Any]) -> str:
    description = props.get("description")
    if description:
        return f"{props.get('name', '')} - {description}"
    return str(props.get("name", ""))


def _build_edge(
    source_id: str, target_id: str, relation: InferredRelation
) -> tuple[str, str, str, dict[str, Any]]:
    return (
        source_id,
        target_id,
        relation.relationship_name,
        {
            "relationship_name": relation.relationship_name,
            "source_node_id": source_id,
            "target_node_id": target_id,
            "inferred": True,
            "confidence": relation.confidence,
            "feedback_weight": INFERRED_EDGE_FEEDBACK_WEIGHT,
        },
    )
