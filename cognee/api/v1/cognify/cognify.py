import asyncio
# import logging
from typing import List, Union
import instructor
from openai import OpenAI
from unstructured.cleaners.core import clean
from unstructured.partition.pdf import partition_pdf
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
from cognee.infrastructure.databases.vector  import CollectionConfig, VectorConfig
from cognee.infrastructure import infrastructure_config

config = Config()
config.load()

aclient = instructor.patch(OpenAI())

USER_ID = "default_user"

async def cognify(datasets: Union[str, List[str]] = None, graphdatamodel: object = None):
    """This function is responsible for the cognitive processing of the content."""

    db = DuckDBAdapter()

    if datasets is None or len(datasets) == 0:
        datasets = db.get_datasets()

    awaitables = []

    if isinstance(datasets, list):
        for dataset in datasets:
            awaitables.append(cognify(dataset))

        graphs = await asyncio.gather(*awaitables)
        return graphs[0]

    files_metadata = db.get_files_metadata(datasets)

    awaitables = []

    await initialize_graph(USER_ID,graphdatamodel)

    for file_metadata in files_metadata:
        with open(file_metadata["file_path"], "rb") as file:
            elements = partition_pdf(file = file, strategy = "fast")
            text = "\n".join(map(lambda element: clean(element.text), elements))

            awaitables.append(process_text(text, file_metadata))

    graphs = await asyncio.gather(*awaitables)

    return graphs[0]

async def process_text(input_text: str, file_metadata: dict):
    print(f"Processing document ({file_metadata['id']})")

    classified_categories = []

    try:
        # Classify the content into categories
        classified_categories = await classify_into_categories(
            input_text,
            "classify_content.txt",
            DefaultContentPrediction
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
            LabeledContent
        )
        file_metadata["content_labels"] = content_labels["content_labels"]
    except Exception as e:
        print(e)
        raise e

    await add_document_node(f"DefaultGraphModel:{USER_ID}", file_metadata)
    print(f"Document ({file_metadata['id']}) categorized: {file_metadata['categories']}")

    cognitive_layers = await content_to_cog_layers(
        classified_categories[0],
        response_model = DefaultCognitiveLayer
    )

    cognitive_layers = [layer_subgroup.name for layer_subgroup in cognitive_layers.cognitive_layers]

    async def generate_graph_per_layer(text_input: str, layers: List[str], response_model: KnowledgeGraph = KnowledgeGraph):
        generate_graphs_awaitables = [generate_graph(text_input, "generate_graph_prompt.txt", {"layer": layer}, response_model) for layer in
                layers]

        return await asyncio.gather(*generate_graphs_awaitables)

    # Run the async function for each set of cognitive layers
    layer_graphs = await generate_graph_per_layer(input_text, cognitive_layers)

    print(f"Document ({file_metadata['id']}) layer graphs created")

    await add_classification_nodes(f"DOCUMENT:{file_metadata['id']}", classified_categories[0])

    await add_summary_nodes(f"DOCUMENT:{file_metadata['id']}", {"summary": file_metadata["summary"]})

    await add_label_nodes(f"DOCUMENT:{file_metadata['id']}", {"content_labels": file_metadata["content_labels"]})

    await append_to_graph(layer_graphs, classified_categories[0])

    print(f"Document ({file_metadata['id']}) layers connected")

    print("Document categories, summaries and metadata are: ", str(classified_categories))

    print("Document metadata is: ", str(file_metadata))

    graph_client = get_graph_client(GraphDBType.NETWORKX)

    await graph_client.load_graph_from_file()

    graph = graph_client.graph

    node_descriptions = await extract_node_descriptions(graph.nodes(data = True))

    nodes_by_layer = await group_nodes_by_layer(node_descriptions)

    unique_layers = nodes_by_layer.keys()

    collection_config = CollectionConfig(
        vector_config = VectorConfig(
            distance = "Cosine",
            size = 3072
        )
    )

    try:
        db_engine = infrastructure_config.get_config()["vector_engine"]

        for layer in unique_layers:
            await db_engine.create_collection(layer, collection_config)
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
