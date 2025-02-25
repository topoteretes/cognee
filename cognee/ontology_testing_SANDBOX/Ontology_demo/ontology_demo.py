import os
import asyncio
import logging
from typing import Type, List, Dict, Tuple

from jinja2.compiler import generate
from owlready2 import get_ontology, Thing, Ontology, ClassConstruct
import difflib
from collections import deque

from utils import (
    setup_logging,
    get_max_chunk_tokens,
    get_datasets,
    get_dataset_data,
    Data,
    run_tasks,
    Task,
    get_default_user,
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
    get_graph_engine,
    DocumentChunk,
    extract_content_graph,
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
    KnowledgeGraph,
    add_data_points,
    prune_data,
    prune_system,
    add,
    generate_edge_name,
    generate_node_id,
    generate_node_name,
    Entity,
    EntityType,
    DataPoint,
)


class OntologyNode(DataPoint):
    name: str
    ontology_origin_type: str


async def main():
    file_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),
        "ontology_test_input",
    )

    await prune_data()
    await prune_system(metadata=True)
    await add(file_path)
    await owl_testing_pipeline()


async def owl_testing_pipeline():
    user = await get_default_user()
    datasets = await get_datasets(user.id)

    ontology_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    ontology_file = os.path.join(ontology_path, "basic_ontology.owl")

    if os.path.exists(ontology_file):
        ontology = get_ontology(ontology_file).load()
        print("Ontology loaded successfully.")
    else:
        print(
            f"Warning: Ontology file not found: {ontology_file}. Proceeding with an empty ontology."
        )
        ontology = get_ontology("http://example.org/empty_ontology")

    ontology_nodes_lookup = {
        "classes": {cls.name.lower().replace(" ", "_").strip(): cls for cls in ontology.classes()},
        "individuals": {
            ind.name.lower().replace(" ", "_").strip(): ind for ind in ontology.individuals()
        },
    }

    for dataset in datasets:
        data_documents: List[Data] = await get_dataset_data(dataset_id=dataset.id)

        tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user=user, permissions=["write"]),
            Task(extract_chunks_from_documents, max_chunk_tokens=get_max_chunk_tokens()),
            Task(
                owl_ontology_merging_layer,
                ontology_lookup=ontology_nodes_lookup,
                ontology_connection=ontology,
                graph_model=KnowledgeGraph,
                task_config={"batch_size": 10},
            ),
        ]

        pipeline_run = run_tasks(tasks, dataset.id, data_documents, "cognify_pipeline")

        async for run_status in pipeline_run:
            print(run_status)


def find_closest_match(name: str, category: str, ontology_nodes_lookup: dict) -> str:
    normalized_name = name.lower().replace(" ", "_").strip()
    possible_matches = list(ontology_nodes_lookup.get(category, {}).keys())

    if normalized_name in possible_matches:
        return normalized_name

    best_match = difflib.get_close_matches(normalized_name, possible_matches, n=1, cutoff=0.8)
    return best_match[0] if best_match else None


def get_ontology_subgraph(
    ontology: Ontology,
    ontology_nodes_lookup: Dict[str, Dict[str, Thing]],
    node_name: str,
    node_type: str = "individuals",
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    nodes = set()
    relationships = []
    visited_nodes = set()
    queue = deque()

    closest_match = find_closest_match(
        name=node_name, category=node_type, ontology_nodes_lookup=ontology_nodes_lookup
    )

    if not closest_match:
        return list(nodes), relationships

    node = ontology_nodes_lookup[node_type].get(closest_match)

    if node is None:
        return list(nodes), relationships

    queue.append(node)
    visited_nodes.add(node)
    nodes.add(node.name)

    while queue:
        current_node = queue.popleft()

        if hasattr(current_node, "is_a"):
            for parent in current_node.is_a:
                if isinstance(parent, ClassConstruct):
                    if hasattr(parent, "value") and hasattr(parent.value, "name"):
                        parent = parent.value
                    else:
                        continue

                relationships.append((current_node.name, "is_a", parent.name))
                nodes.add(parent.name)
                if parent not in visited_nodes:
                    visited_nodes.add(parent)
                    queue.append(parent)

        for prop in ontology.object_properties():
            for target in prop[current_node]:
                relationships.append((current_node.name, prop.name, target.name))
                nodes.add(target.name)
                if target not in visited_nodes:
                    visited_nodes.add(target)
                    queue.append(target)

            for source in prop.range:
                if current_node in prop[source]:
                    relationships.append((source.name, prop.name, current_node.name))
                    nodes.add(source.name)
                    if source not in visited_nodes:
                        visited_nodes.add(source)
                        queue.append(source)

    return list(nodes), relationships


async def owl_ontology_merging_layer(
    data_chunks: List[DocumentChunk],
    ontology_lookup: dict,
    ontology_connection: Ontology,
    graph_model: Type = KnowledgeGraph,
) -> List[DocumentChunk]:
    # We are collecting the LLM generated knowledge graphs generated from DocumentChunks
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    # Collecting nodes and edges from Knowledge graph structures and combining them with Ontologies
    existing_edges_map = {}

    if existing_edges_map is None:
        existing_edges_map = {}

    added_nodes_map = {}
    added_ont_nodes_map = {}

    relationships = []

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

                ontology_entity_type_nodes, ontology_entity_type_edges = get_ontology_subgraph(
                    ontology=ontology_connection,
                    ontology_nodes_lookup=ontology_lookup,
                    node_name=type_node_name,
                    node_type="classes",
                )

                for ont_to_store in ontology_entity_type_nodes:
                    ont_node_id = generate_node_id(ont_to_store)
                    ont_node_name = generate_node_name(ont_to_store)

                    if f"{str(ont_node_id)}_ontology" not in added_ont_nodes_map:
                        ontology_class_node = OntologyNode(
                            id=ont_node_id, name=ont_node_name, ontology_origin_type="class"
                        )
                        added_ont_nodes_map[f"{str(ont_node_id)}_ontology"] = ontology_class_node

                for ont_edge in ontology_entity_type_edges:
                    source_node_id = generate_node_id(ont_edge[0])
                    target_node_id = generate_node_id(ont_edge[2])
                    relationship_name = generate_edge_name(ont_edge[1])

                    edge_key = str(source_node_id) + str(target_node_id) + relationship_name

                    if edge_key not in existing_edges_map:
                        relationships.append(
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

                ontology_entity_nodes, ontology_entity_edges = get_ontology_subgraph(
                    ontology=ontology_connection,
                    ontology_nodes_lookup=ontology_lookup,
                    node_name=node_name,
                    node_type="individuals",
                )

                for ont_to_store in ontology_entity_nodes:
                    ont_node_id = generate_node_id(ont_to_store)
                    ont_node_name = generate_node_name(ont_to_store)

                    if f"{str(ont_node_id)}_ontology" not in added_ont_nodes_map:
                        ontology_node = OntologyNode(
                            id=ont_node_id, name=ont_node_name, ontology_origin_type="individual"
                        )
                        added_ont_nodes_map[f"{str(ont_node_id)}_ontology"] = ontology_node

                for ont_edge in ontology_entity_edges:
                    source_node_id = generate_node_id(ont_edge[0])
                    target_node_id = generate_node_id(ont_edge[2])
                    relationship_name = generate_edge_name(ont_edge[1])

                    edge_key = str(source_node_id) + str(target_node_id) + relationship_name

                    if edge_key not in existing_edges_map:
                        relationships.append(
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

        # Add relationship that came from graphs.
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

    # Adding graph nodes and edges to the database
    graph_nodes = data_chunks + list(added_ont_nodes_map.values())
    graph_edges = relationships

    graph_engine = await get_graph_engine()

    if graph_nodes:
        await add_data_points(graph_nodes)

    if graph_edges:
        await graph_engine.add_edges(graph_edges)

    return data_chunks


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
