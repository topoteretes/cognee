from typing import Optional

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType, Ontology
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_id,
    generate_node_name,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.databases.ontology.rdf_xml.OntologyAdapter import OntologyAdapter


def expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_adapter: OntologyAdapter,
    existing_edges_map: Optional[dict[str, bool]] = None,
):
    if existing_edges_map is None:
        existing_edges_map = {}

    added_nodes_map = {}
    added_ontology_nodes_map = {}

    relationships = []
    ontology_relationships = []

    for index, data_chunk in enumerate(data_chunks):
        graph = chunk_graphs[index]

        if graph is None:
            continue

        for node in graph.nodes:
            node_id = generate_node_id(node.id)
            node_name = generate_node_name(node.name)

            type_node_id = generate_node_id(node.type)
            type_node_name = generate_node_name(node.type)

            if f"{str(type_node_id)}_type" not in added_nodes_map:
                type_node = EntityType(
                    id=type_node_id,
                    name=type_node_name,
                    type=type_node_name,
                    description=type_node_name,
                )
                added_nodes_map[f"{str(type_node_id)}_type"] = type_node

                ontology_entity_type_nodes, ontology_entity_type_edges = (
                    ontology_adapter.get_subgraph(node_name=type_node_name, node_type="classes")
                )

                for ont_to_store in ontology_entity_type_nodes:
                    ont_node_id = generate_node_id(ont_to_store)
                    ont_node_name = generate_node_name(ont_to_store)

                    if f"{str(ont_node_id)}_ontology" not in added_ontology_nodes_map:
                        ontology_class_node = Ontology(
                            id=ont_node_id, name=ont_node_name, ontology_origin_type="class"
                        )
                        added_ontology_nodes_map[f"{str(ont_node_id)}_ontology"] = (
                            ontology_class_node
                        )
                for ont_edge in ontology_entity_type_edges:
                    source_node_id = generate_node_id(ont_edge[0])
                    target_node_id = generate_node_id(ont_edge[2])
                    relationship_name = generate_edge_name(ont_edge[1])

                    edge_key = str(source_node_id) + str(target_node_id) + relationship_name

                    if edge_key not in existing_edges_map:
                        ontology_relationships.append(
                            (
                                source_node_id,
                                target_node_id,
                                relationship_name,
                                dict(
                                    relationship_name=generate_edge_name(relationship_name),
                                    source_node_id=source_node_id,
                                    target_node_id=target_node_id,
                                ),
                            )
                        )
                        existing_edges_map[edge_key] = True
            else:
                type_node = added_nodes_map[f"{str(type_node_id)}_type"]

            if f"{str(node_id)}_entity" not in added_nodes_map:
                entity_node = Entity(
                    id=node_id,
                    name=node_name,
                    is_a=type_node,
                    description=node.description,
                )

                added_nodes_map[f"{str(node_id)}_entity"] = entity_node

                ontology_entity_nodes, ontology_entity_edges = ontology_adapter.get_subgraph(
                    node_name=node_name, node_type="individuals"
                )

                for ont_to_store in ontology_entity_nodes:
                    ont_node_id = generate_node_id(ont_to_store)
                    ont_node_name = generate_node_name(ont_to_store)

                    if f"{str(ont_node_id)}_ontology" not in added_ontology_nodes_map:
                        ontology_node = Ontology(
                            id=ont_node_id, name=ont_node_name, ontology_origin_type="individual"
                        )
                        added_ontology_nodes_map[f"{str(ont_node_id)}_ontology"] = ontology_node

                for ont_edge in ontology_entity_edges:
                    source_node_id = generate_node_id(ont_edge[0])
                    target_node_id = generate_node_id(ont_edge[2])
                    relationship_name = generate_edge_name(ont_edge[1])

                    edge_key = str(source_node_id) + str(target_node_id) + relationship_name

                    if edge_key not in existing_edges_map:
                        ontology_relationships.append(
                            (
                                source_node_id,
                                target_node_id,
                                relationship_name,
                                dict(
                                    relationship_name=generate_edge_name(relationship_name),
                                    source_node_id=source_node_id,
                                    target_node_id=target_node_id,
                                ),
                            )
                        )
                        existing_edges_map[edge_key] = True

            else:
                entity_node = added_nodes_map[f"{str(node_id)}_entity"]

            if data_chunk.contains is None:
                data_chunk.contains = []

            data_chunk.contains.append(entity_node)

        for edge in graph.edges:
            source_node_id = generate_node_id(edge.source_node_id)
            target_node_id = generate_node_id(edge.target_node_id)
            relationship_name = generate_edge_name(edge.relationship_name)

            edge_key = str(source_node_id) + str(target_node_id) + relationship_name

            if edge_key not in existing_edges_map:
                relationships.append(
                    (
                        source_node_id,
                        target_node_id,
                        edge.relationship_name,
                        dict(
                            relationship_name=generate_edge_name(edge.relationship_name),
                            source_node_id=source_node_id,
                            target_node_id=target_node_id,
                        ),
                    )
                )
                existing_edges_map[edge_key] = True

    graph_nodes = data_chunks + list(added_ontology_nodes_map.values())
    graph_edges = relationships + ontology_relationships

    return (graph_nodes, graph_edges)
