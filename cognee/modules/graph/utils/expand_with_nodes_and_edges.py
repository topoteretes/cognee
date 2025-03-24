from typing import Optional

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_id,
    generate_node_name,
)
from owlready2 import Thing, ThingClass
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver


def expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver: OntologyResolver = OntologyResolver(),
    existing_edges_map: Optional[dict[str, bool]] = None,
):
    if existing_edges_map is None:
        existing_edges_map = {}

    added_nodes_map = {}
    added_ontology_nodes_map = {}
    relationships = []
    ontology_relationships = []

    name_mapping = {}
    key_mapping = {}

    for data_chunk, graph in zip(data_chunks, chunk_graphs):
        if not graph:
            continue

        for node in graph.nodes:
            node_id = generate_node_id(node.id)
            node_name = generate_node_name(node.name)
            type_node_id = generate_node_id(node.type)
            type_node_name = generate_node_name(node.type)

            ontology_validated_source_type = False
            ontology_validated_source_ent = False

            type_node_key = f"{type_node_id}_type"

            if type_node_key not in added_nodes_map and type_node_key not in key_mapping:
                (
                    ontology_entity_type_nodes,
                    ontology_entity_type_edges,
                    ontology_closest_class_node,
                ) = ontology_resolver.get_subgraph(node_name=type_node_name, node_type="classes")

                if ontology_closest_class_node:
                    name_mapping[type_node_name] = ontology_closest_class_node.name
                    ontology_validated_source_type = True
                    old_key = type_node_key
                    type_node_id = generate_node_id(ontology_closest_class_node.name)
                    type_node_key = f"{type_node_id}_type"
                    type_node_name = generate_node_name(ontology_closest_class_node.name)
                    key_mapping[old_key] = type_node_key

                type_node = EntityType(
                    id=type_node_id,
                    name=type_node_name,
                    type=type_node_name,
                    description=type_node_name,
                    ontology_valid=ontology_validated_source_type,
                )
                added_nodes_map[type_node_key] = type_node

                for ontology_node_to_store in ontology_entity_type_nodes:
                    ont_node_id = generate_node_id(ontology_node_to_store.name)
                    ont_node_name = generate_node_name(ontology_node_to_store.name)

                    if isinstance(ontology_node_to_store, ThingClass):
                        ont_node_key = f"{ont_node_id}_type"
                        if (ont_node_key not in added_nodes_map) and (
                            ont_node_key not in added_ontology_nodes_map
                        ):
                            added_ontology_nodes_map[ont_node_key] = EntityType(
                                id=ont_node_id,
                                name=ont_node_name,
                                description=ont_node_name,
                                ontology_valid=True,
                            )

                    elif isinstance(ontology_node_to_store, Thing):
                        ont_node_key = f"{ont_node_id}_entity"
                        if (ont_node_key not in added_nodes_map) and (
                            ont_node_key not in added_ontology_nodes_map
                        ):
                            added_ontology_nodes_map[ont_node_key] = Entity(
                                id=ont_node_id,
                                name=ont_node_name,
                                description=ont_node_name,
                                ontology_valid=True,
                            )

                for source, relation, target in ontology_entity_type_edges:
                    source_node_id = generate_node_id(source)
                    target_node_id = generate_node_id(target)
                    relationship_name = generate_edge_name(relation)
                    edge_key = f"{source_node_id}_{target_node_id}_{relationship_name}"

                    if edge_key not in existing_edges_map:
                        ontology_relationships.append(
                            (
                                source_node_id,
                                target_node_id,
                                relationship_name,
                                dict(
                                    relationship_name=relationship_name,
                                    source_node_id=source_node_id,
                                    target_node_id=target_node_id,
                                ),
                            )
                        )
                        existing_edges_map[edge_key] = True
            else:
                type_node = added_nodes_map.get(type_node_key) or added_nodes_map.get(
                    key_mapping.get(type_node_key)
                )

            entity_node_key = f"{node_id}_entity"

            if entity_node_key not in added_nodes_map and entity_node_key not in key_mapping:
                ontology_entity_nodes, ontology_entity_edges, start_ent_ont = (
                    ontology_resolver.get_subgraph(node_name=node_name, node_type="individuals")
                )

                if start_ent_ont:
                    name_mapping[node_name] = start_ent_ont.name
                    ontology_validated_source_ent = True
                    old_key = entity_node_key
                    node_id = generate_node_id(start_ent_ont.name)
                    entity_node_key = f"{node_id}_entity"
                    node_name = generate_node_name(start_ent_ont.name)
                    key_mapping[old_key] = entity_node_key

                entity_node = Entity(
                    id=node_id,
                    name=node_name,
                    is_a=type_node,
                    description=node.description,
                    ontology_valid=ontology_validated_source_ent,
                )

                added_nodes_map[entity_node_key] = entity_node

                for ontology_node_to_store in ontology_entity_nodes:
                    ont_node_id = generate_node_id(ontology_node_to_store.name)
                    ont_node_name = generate_node_name(ontology_node_to_store.name)

                    if isinstance(ontology_node_to_store, ThingClass):
                        ont_node_key = f"{ont_node_id}_type"
                        if (ont_node_key not in added_nodes_map) and (
                            ont_node_key not in added_ontology_nodes_map
                        ):
                            added_ontology_nodes_map[ont_node_key] = EntityType(
                                id=ont_node_id,
                                name=ont_node_name,
                                description=ont_node_name,
                                ontology_valid=True,
                            )

                    elif isinstance(ontology_node_to_store, Thing):
                        ont_node_key = f"{ont_node_id}_entity"
                        if (ont_node_key not in added_nodes_map) and (
                            ont_node_key not in added_ontology_nodes_map
                        ):
                            added_ontology_nodes_map[ont_node_key] = Entity(
                                id=ont_node_id,
                                name=ont_node_name,
                                description=ont_node_name,
                                ontology_valid=True,
                            )

                for source, relation, target in ontology_entity_edges:
                    source_node_id = generate_node_id(source)
                    target_node_id = generate_node_id(target)
                    relationship_name = generate_edge_name(relation)
                    edge_key = f"{source_node_id}_{target_node_id}_{relationship_name}"

                    if edge_key not in existing_edges_map:
                        ontology_relationships.append(
                            (
                                source_node_id,
                                target_node_id,
                                relationship_name,
                                dict(
                                    relationship_name=relationship_name,
                                    source_node_id=source_node_id,
                                    target_node_id=target_node_id,
                                    ontology_valid=True,
                                ),
                            )
                        )
                        existing_edges_map[edge_key] = True

            else:
                entity_node = added_nodes_map.get(entity_node_key) or added_nodes_map.get(
                    key_mapping.get(entity_node_key)
                )

            if data_chunk.contains is None:
                data_chunk.contains = []

            data_chunk.contains.append(entity_node)

        for edge in graph.edges:
            source_node_id = generate_node_id(
                name_mapping.get(edge.source_node_id, edge.source_node_id)
            )
            target_node_id = generate_node_id(
                name_mapping.get(edge.target_node_id, edge.target_node_id)
            )
            relationship_name = generate_edge_name(edge.relationship_name)
            edge_key = f"{source_node_id}_{target_node_id}_{relationship_name}"

            if edge_key not in existing_edges_map:
                relationships.append(
                    (
                        source_node_id,
                        target_node_id,
                        relationship_name,
                        dict(
                            relationship_name=relationship_name,
                            source_node_id=source_node_id,
                            target_node_id=target_node_id,
                            ontology_valid=False,
                        ),
                    )
                )
                existing_edges_map[edge_key] = True

    graph_nodes = data_chunks + list(added_ontology_nodes_map.values())
    graph_edges = relationships + ontology_relationships

    return graph_nodes, graph_edges
