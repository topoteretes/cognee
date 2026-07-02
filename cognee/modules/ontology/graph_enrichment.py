from dataclasses import dataclass
from typing import Optional

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


def _create_node_key(node_id: str, category: str) -> str:
    """Create a standardized node key."""
    return f"{node_id}_{category}"


def _create_edge_key(source_id: str, target_id: str, relationship_name: str) -> str:
    """Create a standardized edge key."""
    return f"{source_id}_{target_id}_{relationship_name}"


@dataclass(frozen=True)
class OntologyMatch:
    category: str
    canonical_name: str
    subgraph_nodes: list
    subgraph_edges: list
    triggering_chunk: DocumentChunk


def _resolve_ontology_match(
    ontology_resolver: BaseOntologyResolver,
    node_name: str,
    node_type: str,
) -> tuple[Optional[str], list, list]:
    """Return (canonical_raw_name, ontology_nodes, ontology_edges)."""
    ontology_nodes, ontology_edges, match = ontology_resolver.get_subgraph(
        node_name=node_name, node_type=node_type
    )
    if not match:
        return None, [], []
    return match.name, ontology_nodes, ontology_edges


def canonicalize_graphs(
    chunk_graphs: list[KnowledgeGraph],
    data_chunks: list[DocumentChunk],
    ontology_resolver: Optional[BaseOntologyResolver],
) -> tuple[list[KnowledgeGraph], list[OntologyMatch]]:
    """Rewrite extracted graphs to canonical ontology names before conversion."""
    if ontology_resolver is None:
        return chunk_graphs, []

    resolved_types: dict[str, tuple[Optional[str], list, list]] = {}
    resolved_entities: dict[str, tuple[Optional[str], list, list]] = {}
    matches: list[OntologyMatch] = []

    for chunk, graph in zip(data_chunks, chunk_graphs):
        if not graph:
            continue
        for node in graph.nodes:
            norm_type = generate_node_name(node.type)
            if norm_type not in resolved_types:
                resolved_types[norm_type] = _resolve_ontology_match(
                    ontology_resolver, norm_type, "classes"
                )
                canonical_name, ontology_nodes, ontology_edges = resolved_types[norm_type]
                if canonical_name is not None:
                    matches.append(
                        OntologyMatch(
                            category="classes",
                            canonical_name=canonical_name,
                            subgraph_nodes=ontology_nodes,
                            subgraph_edges=ontology_edges,
                            triggering_chunk=chunk,
                        )
                    )

            norm_name = generate_node_name(node.name)
            if norm_name not in resolved_entities:
                resolved_entities[norm_name] = _resolve_ontology_match(
                    ontology_resolver, norm_name, "individuals"
                )
                canonical_name, ontology_nodes, ontology_edges = resolved_entities[norm_name]
                if canonical_name is not None:
                    matches.append(
                        OntologyMatch(
                            category="individuals",
                            canonical_name=canonical_name,
                            subgraph_nodes=ontology_nodes,
                            subgraph_edges=ontology_edges,
                            triggering_chunk=chunk,
                        )
                    )

    endpoint_rewrite: dict[str, str] = {}
    for graph in chunk_graphs:
        if not graph:
            continue
        for node in graph.nodes:
            norm_type = generate_node_name(node.type)
            canonical_type = resolved_types[norm_type][0]
            if canonical_type is not None:
                endpoint_rewrite[norm_type] = canonical_type

            norm_name = generate_node_name(node.name)
            norm_id = generate_node_name(node.id)
            canonical_entity = resolved_entities[norm_name][0]
            if canonical_entity is not None:
                endpoint_rewrite[norm_name] = canonical_entity
                endpoint_rewrite[norm_id] = canonical_entity

    def rewrite_endpoint(endpoint: str) -> str:
        return endpoint_rewrite.get(generate_node_name(endpoint), endpoint)

    copied_graphs: list[KnowledgeGraph] = []
    for graph in chunk_graphs:
        if not graph:
            copied_graphs.append(graph)
            continue

        copied_nodes = []
        for node in graph.nodes:
            norm_type = generate_node_name(node.type)
            canonical_type = resolved_types[norm_type][0]
            new_type = canonical_type if canonical_type is not None else node.type

            norm_name = generate_node_name(node.name)
            canonical_entity = resolved_entities[norm_name][0]
            if canonical_entity is not None:
                new_id = canonical_entity
                new_name = canonical_entity
            else:
                new_id = node.id
                new_name = node.name

            copied_nodes.append(
                Node(id=new_id, name=new_name, type=new_type, description=node.description)
            )

        copied_edges = [
            KGEdge(
                source_node_id=rewrite_endpoint(edge.source_node_id),
                target_node_id=rewrite_endpoint(edge.target_node_id),
                relationship_name=edge.relationship_name,
                description=edge.description,
            )
            for edge in graph.edges
        ]
        copied_graphs.append(KnowledgeGraph(nodes=copied_nodes, edges=copied_edges))

    return copied_graphs, matches


def _mark_matched_nodes_valid(matches: list[OntologyMatch], added_nodes_map: dict) -> None:
    """Flip ontology_valid on converted nodes that canonicalization matched."""
    for match in matches:
        if match.category == "classes":
            node_key = _create_node_key(EntityType.id_for(match.canonical_name), "type")
        else:
            node_key = _create_node_key(Entity.id_for(match.canonical_name), "entity")
        node = added_nodes_map.get(node_key)
        if node is not None:
            node.ontology_valid = True


def _process_ontology_nodes(
    ontology_nodes: list,
    data_chunk: DocumentChunk,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
) -> None:
    """Process and store ontology nodes."""
    for ontology_node in ontology_nodes:
        ont_node_id = (
            EntityType.id_for(ontology_node.name)
            if ontology_node.category == "classes"
            else Entity.id_for(ontology_node.name)
        )
        ont_node_name = generate_node_name(ontology_node.name)

        if ontology_node.category == "classes":
            ont_node_key = _create_node_key(ont_node_id, "type")
            if ont_node_key not in added_nodes_map and ont_node_key not in added_ontology_nodes_map:
                added_ontology_nodes_map[ont_node_key] = EntityType(
                    id=ont_node_id,
                    name=ont_node_name,
                    description=ont_node_name,
                    ontology_valid=True,
                    importance_weight=data_chunk.importance_weight,
                )

        elif ontology_node.category == "individuals":
            ont_node_key = _create_node_key(ont_node_id, "entity")
            if ont_node_key not in added_nodes_map and ont_node_key not in added_ontology_nodes_map:
                added_ontology_nodes_map[ont_node_key] = Entity(
                    id=ont_node_id,
                    name=ont_node_name,
                    description=ont_node_name,
                    ontology_valid=True,
                    belongs_to_set=data_chunk.belongs_to_set,
                    importance_weight=data_chunk.importance_weight,
                )


def _process_ontology_edges(
    ontology_nodes: list,
    ontology_edges: list,
    existing_edges_map: dict,
    ontology_relationships: list,
) -> None:
    """Process ontology edges and add them if new."""
    node_category = {node.name: node.category for node in ontology_nodes}
    for source, relation, target in ontology_edges:
        source_cls = EntityType if node_category.get(source) == "classes" else Entity
        target_cls = EntityType if node_category.get(target) == "classes" else Entity
        source_node_id = source_cls.id_for(source)
        target_node_id = target_cls.id_for(target)
        relationship_name = generate_edge_name(relation)
        edge_key = _create_edge_key(source_node_id, target_node_id, relationship_name)

        if edge_key not in existing_edges_map:
            ontology_relationships.append(
                (
                    source_node_id,
                    target_node_id,
                    relationship_name,
                    {
                        "relationship_name": relationship_name,
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "ontology_valid": True,
                    },
                )
            )
            existing_edges_map[edge_key] = True


def extend_graph_with_ontology(
    matches: list[OntologyMatch],
    added_nodes_map: dict,
    existing_edges_map: dict,
) -> tuple[dict, list]:
    """Materialize matched subgraphs after conversion, deduped against the full node set."""
    added_ontology_nodes_map = {}
    ontology_relationships = []

    _mark_matched_nodes_valid(matches, added_nodes_map)

    for match in matches:
        _process_ontology_nodes(
            match.subgraph_nodes,
            match.triggering_chunk,
            added_nodes_map,
            added_ontology_nodes_map,
        )
        _process_ontology_edges(
            match.subgraph_nodes,
            match.subgraph_edges,
            existing_edges_map,
            ontology_relationships,
        )

    return added_ontology_nodes_map, ontology_relationships
