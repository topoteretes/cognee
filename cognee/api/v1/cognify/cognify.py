import asyncio
# import logging
from typing import List
from qdrant_client import models
import instructor
from openai import OpenAI
from unstructured.cleaners.core import clean
from unstructured.partition.pdf import partition_pdf
from cognee.infrastructure.databases.vector.qdrant.QDrantAdapter import CollectionConfig
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.cognify.graph.add_classification_nodes import add_classification_nodes
from cognee.modules.cognify.graph.add_node_connections import add_node_connection, graph_ready_output, \
    connect_nodes_in_graph, extract_node_descriptions
from cognee.modules.cognify.graph.add_propositions import append_to_graph
from cognee.modules.cognify.llm.add_node_connection_embeddings import process_items
from cognee.modules.cognify.vector.batch_search import adapted_qdrant_batch_search
from cognee.modules.cognify.vector.add_propositions import add_propositions

from cognee.config import Config
from cognee.modules.cognify.llm.classify_content import classify_into_categories
from cognee.modules.cognify.llm.content_to_cog_layers import content_to_cog_layers
from cognee.modules.cognify.llm.generate_graph import generate_graph
from cognee.shared.data_models import DefaultContentPrediction,  KnowledgeGraph, DefaultCognitiveLayer
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType
from cognee.infrastructure.databases.vector.get_vector_database import get_vector_database
from cognee.infrastructure.databases.relational import DuckDBAdapter
from cognee.modules.cognify.graph.add_document_node import add_document_node
from cognee.modules.cognify.graph.initialize_graph import initialize_graph

config = Config()
config.load()

aclient = instructor.patch(OpenAI())

USER_ID = "default_user"

async def cognify(dataset_name: str):
    """This function is responsible for the cognitive processing of the content."""

    db = DuckDBAdapter()
    files_metadata = db.get_files_metadata(dataset_name)

    awaitables = []

    await initialize_graph(USER_ID)

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

    await add_document_node(f"DefaultGraphModel:{USER_ID}", file_metadata)
    print(f"Document ({file_metadata['id']}) categorized: {file_metadata['categories']}")

    cognitive_layers = await content_to_cog_layers(
        "generate_cog_layers.txt",
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
    # print(layer_graphs)

    print(f"Document ({file_metadata['id']}) layer graphs created")

    # G = await create_semantic_graph(graph_model_instance)

    await add_classification_nodes(f"DOCUMENT:{file_metadata['id']}", classified_categories[0])

    unique_layer_uuids = await append_to_graph(layer_graphs, classified_categories[0])

    print(f"Document ({file_metadata['id']}) layers connected")

    graph_client = get_graph_client(GraphDBType.NETWORKX)

    await graph_client.load_graph_from_file()

    graph = graph_client.graph

    # # Extract the node descriptions
    node_descriptions = await extract_node_descriptions(graph.nodes(data = True))
    # print(node_descriptions)

    unique_layer_uuids = set(node["layer_decomposition_uuid"] for node in node_descriptions)

    collection_config = CollectionConfig(
        vector_config = {
            "content": models.VectorParams(
                distance = models.Distance.COSINE,
                size = 3072
            )
        },
    )

    try:
        for layer in unique_layer_uuids:
            db = get_vector_database()
            await db.create_collection(layer, collection_config)
    except Exception as e:
        print(e)

    await add_propositions(node_descriptions)

    grouped_data = await add_node_connection(node_descriptions)
    # print("we are here, grouped_data", grouped_data)

    llm_client = get_llm_client()

    relationship_dict = await process_items(grouped_data, unique_layer_uuids, llm_client)
    # print("we are here", relationship_dict[0])

    results = await adapted_qdrant_batch_search(relationship_dict, db)
    # print(results)

    relationship_d = graph_ready_output(results)
    # print(relationship_d)

    connect_nodes_in_graph(graph, relationship_d)

    print(f"Document ({file_metadata['id']}) processed")

    return graph



if __name__ == "__main__":
    asyncio.run(cognify("""In the nicest possible way, Britons have always been a bit silly about animals. “Keeping pets, for the English, is not so much a leisure activity as it is an entire way of life,” wrote the anthropologist Kate Fox in Watching the English, nearly 20 years ago. Our dogs, in particular, have been an acceptable outlet for emotions and impulses we otherwise keep strictly controlled – our latent desire to be demonstratively affectionate, to be silly and chat to strangers. If this seems like an exaggeration, consider the different reactions you’d get if you struck up a conversation with someone in a park with a dog, versus someone on the train.
Indeed, British society has been set up to accommodate these four-legged ambassadors. In the UK – unlike Australia, say, or New Zealand – dogs are not just permitted on public transport but often openly encouraged. Many pubs and shops display waggish signs, reading, “Dogs welcome, people tolerated”, and have treat jars on their counters. The other day, as I was waiting outside a cafe with a friend’s dog, the barista urged me to bring her inside.
For years, Britons’ non-partisan passion for animals has been consistent amid dwindling common ground. But lately, rather than bringing out the best in us, our relationship with dogs is increasingly revealing us at our worst – and our supposed “best friends” are paying the price.
As with so many latent traits in the national psyche, it all came unleashed with the pandemic, when many people thought they might as well make the most of all that time at home and in local parks with a dog. Between 2019 and 2022, the number of pet dogs in the UK rose from about nine million to 13 million. But there’s long been a seasonal surge around this time of year, substantial enough for the Dogs Trust charity to coin its famous slogan back in 1978: “A dog is for life, not just for Christmas.”
"""))