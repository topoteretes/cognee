from dataclasses import dataclass

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name, generate_node_name


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
