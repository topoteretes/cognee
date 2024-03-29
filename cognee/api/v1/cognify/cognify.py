import asyncio
# import logging
from typing import List, Union
import instructor
from openai import OpenAI
from cognee.modules.cognify.graph.add_classification_nodes import add_classification_nodes
from cognee.modules.cognify.llm.label_content import label_content
from cognee.modules.cognify.graph.add_label_nodes import add_label_nodes
from cognee.modules.cognify.llm.summarize_content import summarize_content
from cognee.modules.cognify.graph.add_summary_nodes import add_summary_nodes
from cognee.modules.cognify.graph.add_node_connections import group_nodes_by_layer, graph_ready_output, \
    connect_nodes_in_graph, extract_node_descriptions
from cognee.modules.cognify.graph.add_propositions import append_to_graph
from cognee.modules.cognify.llm.resolve_cross_graph_references import resolve_cross_graph_references
from cognee.modules.cognify.vector.add_propositions import add_propositions

from cognee.config import Config
from cognee.modules.cognify.llm.classify_content import classify_into_categories
from cognee.modules.cognify.llm.content_to_cog_layers import content_to_cog_layers
from cognee.modules.cognify.llm.generate_graph import generate_graph
from cognee.shared.data_models import DefaultContentPrediction, KnowledgeGraph, DefaultCognitiveLayer, \
    SummarizedContent, LabeledContent
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType
from cognee.infrastructure.databases.relational import DuckDBAdapter
from cognee.modules.cognify.graph.add_document_node import add_document_node
from cognee.modules.cognify.graph.initialize_graph import initialize_graph
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type
from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file
from cognee.infrastructure import infrastructure_config

config = Config()
config.load()

aclient = instructor.patch(OpenAI())

USER_ID = "default_user"

async def cognify(datasets: Union[str, List[str]] = None, graph_data_model: object = None, classification_model: object = None, summarization_model: object = None, labeling_model: object = None, graph_model: object = None, cognitive_layer_model: object = None, graph_db_type: object = None):
    """This function is responsible for the cognitive processing of the content."""

    db = DuckDBAdapter()

    if datasets is None or len(datasets) == 0:
        datasets = db.get_datasets()

    awaitables = []

    # datasets is a list of dataset names
    if isinstance(datasets, list):
        for dataset in datasets:
            awaitables.append(cognify(dataset))

        graphs = await asyncio.gather(*awaitables)
        return graphs[0]

    # datasets is a dataset name string
    added_datasets = db.get_datasets()

    files_metadata = []
    dataset_name = datasets.replace(".", "_").replace(" ", "_")

    for added_dataset in added_datasets:
        if dataset_name in added_dataset:
            files_metadata.extend(db.get_files_metadata(added_dataset))

    awaitables = []

    if graph_db_type is None:
        graph_db_type = GraphDBType.NETWORKX

    graph_client = await get_graph_client(graph_db_type)

    await initialize_graph(USER_ID, graph_data_model, graph_client)

    for file_metadata in files_metadata:
        with open(file_metadata["file_path"], "rb") as file:
            file_type = guess_file_type(file)
            text = extract_text_from_file(file, file_type)

            awaitables.append(process_text(text, file_metadata, graph_data_model, classification_model, summarization_model, labeling_model, graph_model, cognitive_layer_model, graph_db_type))

    graphs = await asyncio.gather(*awaitables)

    return graphs[0]

async def process_text(input_text: str, file_metadata: dict, graph_data_model: object=None, classification_model: object=None, summarization_model: object=None, labeling_model: object=None, graph_model: object=None, cognitive_layer_model: object=None, graph_db_type: object=None):
    print(f"Processing document ({file_metadata['id']})")

    classified_categories = []

    if classification_model is None:
        classification_model= DefaultContentPrediction

    if summarization_model is None:
        summarization_model = SummarizedContent
    if labeling_model is None:
        labeling_model = LabeledContent
    if cognitive_layer_model is None:
        cognitive_layer_model = DefaultCognitiveLayer
    if graph_model is None:
        graph_model = KnowledgeGraph

    if graph_db_type is None:
        graph_db_type = GraphDBType.NETWORKX

    try:
        # Classify the content into categories
        classified_categories = await classify_into_categories(
            input_text,
            "classify_content.txt",
            classification_model
        )
        file_metadata["categories"] = list(map(lambda category: category["layer_name"], classified_categories))
    except Exception as e:
        print(e)
        raise e

    try:
        # Classify the content into categories
        content_summary = await summarize_content(
            input_text,
            "summarize_content.txt",
            SummarizedContent
        )
        file_metadata["summary"] = content_summary["summary"]
    except Exception as e:
        print(e)
        raise e

    try:
        # Classify the content into categories
        content_labels = await label_content(
            input_text,
            "label_content.txt",
            labeling_model
        )
        file_metadata["content_labels"] = content_labels["content_labels"]
    except Exception as e:
        print(e)
        raise e
    graph_client = await get_graph_client(graph_db_type)
    await add_document_node(graph_client, f"DefaultGraphModel:{USER_ID}", file_metadata)
    print(f"Document ({file_metadata['id']}) categorized: {file_metadata['categories']}")

    cognitive_layers = await content_to_cog_layers(
        classified_categories[0],
        response_model = cognitive_layer_model
    )

    cognitive_layers = [layer_subgroup.name for layer_subgroup in cognitive_layers.cognitive_layers]

    async def generate_graph_per_layer(text_input: str, layers: List[str], response_model: KnowledgeGraph = graph_model):
        generate_graphs_awaitables = [generate_graph(text_input, "generate_graph_prompt.txt", {"layer": layer}, response_model) for layer in
                layers]

        return await asyncio.gather(*generate_graphs_awaitables)

    # Run the async function for each set of cognitive layers
    layer_graphs = await generate_graph_per_layer(input_text, cognitive_layers)

    print(f"Document ({file_metadata['id']}) layer graphs created")


    await add_classification_nodes(graph_client,f"DOCUMENT:{file_metadata['id']}", classified_categories[0])

    await add_summary_nodes(graph_client,f"DOCUMENT:{file_metadata['id']}", {"summary": file_metadata["summary"]})

    await add_label_nodes(graph_client,f"DOCUMENT:{file_metadata['id']}", {"content_labels": file_metadata["content_labels"]})

    await append_to_graph(graph_client, layer_graphs, classified_categories[0])

    print(f"Document ({file_metadata['id']}) layers connected")

    print("Document categories, summaries and metadata are: ", str(classified_categories))

    print("Document metadata is: ", str(file_metadata))



    # await graph_client.load_graph_from_file()

    graph = graph_client.graph

    node_descriptions = await extract_node_descriptions(graph.nodes(data = True))

    nodes_by_layer = await group_nodes_by_layer(node_descriptions)

    unique_layers = nodes_by_layer.keys()

    try:
        db_engine = infrastructure_config.get_config()["vector_engine"]

        for layer in unique_layers:
            await db_engine.create_collection(layer)
    except Exception as e:
        print(e)

    await add_propositions(nodes_by_layer)

    results = await resolve_cross_graph_references(nodes_by_layer)

    relationships = graph_ready_output(results)
    # print(relationships)
    await graph_client.load_graph_from_file()

    graph = graph_client.graph

    connect_nodes_in_graph(graph, relationships)

    print(f"Document ({file_metadata['id']}) processed")

    return graph



if __name__ == "__main__":

    async def main():
        graph = await cognify(datasets=['izmene'])
        from cognee.utils import render_graph
        graph_url = await render_graph(graph, graph_type="networkx")
        print(graph_url)


    asyncio.run(main())