from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import construct_data_points_and_edges
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.shared.data_models import KnowledgeGraph, Node

_ONTOLOGY_CLASS_CATEGORY = "classes"
_ONTOLOGY_INDIVIDUAL_CATEGORY = "individuals"


@dataclass(frozen=True)
class OntologyMatch:
    node_category: str
    canonical_name: str
    canonical_uri: Optional[str]
    ontology_nodes: list[AttachedOntologyNode]
    ontology_edges: list[tuple[str, str, str]]
    first_source_chunk: DocumentChunk


OntologyMatchLookup = dict[tuple[str, str], Optional[OntologyMatch]]


def _find_ontology_match(
    ontology_resolver: BaseOntologyResolver,
    normalized_extracted_name: str,
    node_category: str,
    first_source_chunk: DocumentChunk,
) -> Optional[OntologyMatch]:
    """Find the ontology node and subgraph matching one extracted name."""
    ontology_nodes, ontology_edges, matched_ontology_node = ontology_resolver.get_subgraph(
        node_name=normalized_extracted_name,
        node_type=node_category,
    )
    if not matched_ontology_node:
        return None

    return OntologyMatch(
        node_category=node_category,
        canonical_name=matched_ontology_node.name,
        canonical_uri=(
            str(matched_ontology_node.uri) if matched_ontology_node.uri is not None else None
        ),
        ontology_nodes=ontology_nodes,
        ontology_edges=ontology_edges,
        first_source_chunk=first_source_chunk,
    )


def _find_ontology_matches_for_extracted_graphs(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
) -> OntologyMatchLookup:
    """Find one ontology match for each distinct extracted name and type."""
    ontology_match_lookup: OntologyMatchLookup = {}
    for data_chunk, extracted_graph in zip(data_chunks, extracted_graphs):
        if not extracted_graph:
            continue

        for node in extracted_graph.nodes:
            for node_category, extracted_name in (
                (_ONTOLOGY_CLASS_CATEGORY, node.type),
                (_ONTOLOGY_INDIVIDUAL_CATEGORY, node.name),
            ):
                normalized_extracted_name = generate_node_name(extracted_name)
                lookup_key = (node_category, normalized_extracted_name)
                if lookup_key in ontology_match_lookup:
                    continue

                ontology_match_lookup[lookup_key] = _find_ontology_match(
                    ontology_resolver,
                    normalized_extracted_name,
                    node_category,
                    data_chunk,
                )

    return ontology_match_lookup


def _get_ontology_match(
    ontology_match_lookup: OntologyMatchLookup,
    node_category: str,
    extracted_name: str,
) -> Optional[OntologyMatch]:
    """Return the ontology match already looked up for an extracted name."""
    return ontology_match_lookup.get((node_category, generate_node_name(extracted_name)))


def _canonicalize_extracted_graph(
    extracted_graph: KnowledgeGraph,
    ontology_match_lookup: OntologyMatchLookup,
) -> None:
    """Canonicalize one graph and collapse nodes matched to the same ontology entity."""
    extracted_node_ids: set[str] = set()
    for node in extracted_graph.nodes:
        if node.id in extracted_node_ids:
            raise ValueError(f"Duplicate node id in extracted graph: {node.id}")
        extracted_node_ids.add(node.id)

    retained_nodes: list[Node] = []
    surviving_node_id_by_entity_id: dict[UUID, str] = {}
    surviving_node_id_by_collapsed_node_id: dict[str, str] = {}

    for node in extracted_graph.nodes:
        entity_type_match = _get_ontology_match(
            ontology_match_lookup,
            _ONTOLOGY_CLASS_CATEGORY,
            node.type,
        )
        if entity_type_match is not None:
            node.type = entity_type_match.canonical_name

        entity_match = _get_ontology_match(
            ontology_match_lookup,
            _ONTOLOGY_INDIVIDUAL_CATEGORY,
            node.name,
        )
        if entity_match is None:
            retained_nodes.append(node)
            continue

        node.name = entity_match.canonical_name
        canonical_entity_id = Entity.id_for(entity_match.canonical_name)
        surviving_node_id = surviving_node_id_by_entity_id.get(canonical_entity_id)
        if surviving_node_id is None:
            surviving_node_id_by_entity_id[canonical_entity_id] = node.id
            retained_nodes.append(node)
            continue

        surviving_node_id_by_collapsed_node_id[node.id] = surviving_node_id

    extracted_graph.nodes = retained_nodes
    if not surviving_node_id_by_collapsed_node_id:
        return

    for edge in extracted_graph.edges:
        edge.source_node_id = surviving_node_id_by_collapsed_node_id.get(
            edge.source_node_id,
            edge.source_node_id,
        )
        edge.target_node_id = surviving_node_id_by_collapsed_node_id.get(
            edge.target_node_id,
            edge.target_node_id,
        )


def canonicalize_extracted_graphs(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
) -> tuple[list[KnowledgeGraph], OntologyMatchLookup]:
    """Canonicalize ontology matches in place before ordinary graph construction."""
    ontology_match_lookup = _find_ontology_matches_for_extracted_graphs(
        data_chunks,
        extracted_graphs,
        ontology_resolver,
    )

    for extracted_graph in extracted_graphs:
        if extracted_graph:
            _canonicalize_extracted_graph(extracted_graph, ontology_match_lookup)

    return extracted_graphs, ontology_match_lookup


def _get_data_point_class_for_ontology_category(
    node_category: str,
) -> type[Entity] | type[EntityType] | None:
    if node_category == _ONTOLOGY_CLASS_CATEGORY:
        return EntityType
    if node_category == _ONTOLOGY_INDIVIDUAL_CATEGORY:
        return Entity
    return None


def _mark_existing_ontology_data_point(
    data_point_class: type[Entity] | type[EntityType],
    ontology_name: str,
    ontology_uri: Optional[str],
    data_points_by_id: dict[str, Entity | EntityType],
) -> bool:
    data_point_id = data_point_class.id_for(ontology_name)
    data_point = data_points_by_id.get(str(data_point_id))
    if data_point is None:
        return False

    data_point.ontology_valid = True
    if ontology_uri is not None:
        data_point.ontology_uri = ontology_uri
    return True


def _add_ontology_data_points(
    ontology_nodes: list[AttachedOntologyNode],
    source_chunk: DocumentChunk,
    data_points_by_id: dict[str, Entity | EntityType],
) -> None:
    for ontology_node in ontology_nodes:
        data_point_class = _get_data_point_class_for_ontology_category(ontology_node.category)
        if data_point_class is None:
            continue

        ontology_uri = str(ontology_node.uri) if ontology_node.uri is not None else None
        if _mark_existing_ontology_data_point(
            data_point_class,
            ontology_node.name,
            ontology_uri,
            data_points_by_id,
        ):
            continue

        data_point_id = data_point_class.id_for(ontology_node.name)
        data_point_key = str(data_point_id)
        normalized_name = generate_node_name(ontology_node.name)
        if data_point_class is EntityType:
            data_points_by_id[data_point_key] = EntityType(
                id=data_point_id,
                name=normalized_name,
                description=normalized_name,
                ontology_valid=True,
                ontology_uri=ontology_uri,
                importance_weight=source_chunk.importance_weight,
            )
        else:
            data_points_by_id[data_point_key] = Entity(
                id=data_point_id,
                name=normalized_name,
                description=normalized_name,
                ontology_valid=True,
                ontology_uri=ontology_uri,
                belongs_to_set=source_chunk.belongs_to_set,
                importance_weight=source_chunk.importance_weight,
            )


def _add_ontology_edges(
    ontology_nodes: list[AttachedOntologyNode],
    ontology_edges: list[tuple[str, str, str]],
    edges_by_identity: dict[EdgeIdentity, Edge],
) -> None:
    ontology_nodes_by_normalized_name = {
        generate_edge_name(node.name): node for node in ontology_nodes
    }
    for source_name, ontology_relationship_name, target_name in ontology_edges:
        source_node = ontology_nodes_by_normalized_name.get(generate_edge_name(source_name))
        target_node = ontology_nodes_by_normalized_name.get(generate_edge_name(target_name))
        if source_node is None or target_node is None:
            continue

        source_data_point_class = _get_data_point_class_for_ontology_category(source_node.category)
        target_data_point_class = _get_data_point_class_for_ontology_category(target_node.category)
        if source_data_point_class is None or target_data_point_class is None:
            continue

        relationship_name = generate_edge_name(ontology_relationship_name)
        edge_identity = EdgeIdentity(
            source_id=str(source_data_point_class.id_for(source_node.name)),
            target_id=str(target_data_point_class.id_for(target_node.name)),
            relationship_name=relationship_name,
        )
        edges_by_identity.setdefault(
            edge_identity,
            Edge(relationship_type=relationship_name),
        )


def add_ontology_data_points_and_edges(
    ontology_match_lookup: OntologyMatchLookup,
    data_points_by_id: dict[str, Entity | EntityType],
    edges_by_identity: dict[EdgeIdentity, Edge],
) -> None:
    """Add the nodes and edges from each uniquely matched ontology subgraph."""
    added_ontology_root_keys: set[tuple[str, str]] = set()
    for ontology_match in ontology_match_lookup.values():
        if ontology_match is None:
            continue

        ontology_root_key = (
            ontology_match.node_category,
            ontology_match.canonical_uri or ontology_match.canonical_name,
        )
        if ontology_root_key in added_ontology_root_keys:
            continue
        added_ontology_root_keys.add(ontology_root_key)

        canonical_data_point_class = _get_data_point_class_for_ontology_category(
            ontology_match.node_category
        )
        if canonical_data_point_class is not None:
            _mark_existing_ontology_data_point(
                canonical_data_point_class,
                ontology_match.canonical_name,
                ontology_match.canonical_uri,
                data_points_by_id,
            )

        _add_ontology_data_points(
            ontology_match.ontology_nodes,
            ontology_match.first_source_chunk,
            data_points_by_id,
        )
        _add_ontology_edges(
            ontology_match.ontology_nodes,
            ontology_match.ontology_edges,
            edges_by_identity,
        )


def construct_data_points_and_edges_with_ontology(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
) -> tuple[dict[str, Entity | EntityType], dict[EdgeIdentity, Edge]]:
    """Canonicalize, construct, and enrich extracted graphs with ontology data."""
    canonicalized_graphs, ontology_match_lookup = canonicalize_extracted_graphs(
        data_chunks,
        extracted_graphs,
        ontology_resolver,
    )
    data_points_by_id, edges_by_identity = construct_data_points_and_edges(
        data_chunks,
        canonicalized_graphs,
    )
    add_ontology_data_points_and_edges(
        ontology_match_lookup,
        data_points_by_id,
        edges_by_identity,
    )
    return data_points_by_id, edges_by_identity
