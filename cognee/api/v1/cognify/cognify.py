import asyncio
from typing import List, Union
import instructor
import logging
from openai import OpenAI
from cognee.modules.cognify.graph.add_classification_nodes import add_classification_nodes
from cognee.modules.cognify.graph.add_cognitive_layer_graphs import add_cognitive_layer_graphs
# from cognee.modules.cognify.llm.label_content import label_content
# from cognee.modules.cognify.graph.add_label_nodes import add_label_nodes
from cognee.modules.cognify.graph.add_summary_nodes import add_summary_nodes
# from cognee.modules.cognify.graph.add_node_connections import group_nodes_by_layer, graph_ready_output, \
    # connect_nodes_in_graph
# from cognee.modules.cognify.graph.add_propositions import append_to_graph
# from cognee.modules.cognify.llm.resolve_cross_graph_references import resolve_cross_graph_references
# from cognee.modules.cognify.vector.add_propositions import add_propositions

from cognee.config import Config

# from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.modules.cognify.graph.add_document_node import add_document_node
from cognee.modules.cognify.graph.add_keywords_nodes import add_keywords_nodes
from cognee.modules.cognify.graph.add_cognitive_layers import add_cognitive_layers
from cognee.modules.cognify.graph.initialize_graph import initialize_graph
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type, FileTypeException
from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file
from cognee.infrastructure import infrastructure_config
from cognee.modules.data.get_content_categories import get_content_categories
from cognee.modules.data.get_content_summary import get_content_summary
from cognee.modules.data.get_cognitive_layers import get_cognitive_layers
from cognee.modules.data.get_layer_graphs import get_layer_graphs
from cognee.shared.data_models import GraphDBType

config = Config()
config.load()

aclient = instructor.patch(OpenAI())

USER_ID = "default_user"

logger = logging.getLogger(__name__)

async def cognify(datasets: Union[str, List[str]] = None, graph_data_model: object = None):
    """This function is responsible for the cognitive processing of the content."""

    db_engine = infrastructure_config.get_config()["database_engine"]

    if datasets is None or len(datasets) == 0:
        datasets = db_engine.get_datasets()

    awaitables = []

    # datasets is a list of dataset names
    if isinstance(datasets, list):
        for dataset in datasets:
            awaitables.append(cognify(dataset))

        graphs = await asyncio.gather(*awaitables)
        return graphs[0]

    # datasets is a dataset name string
    added_datasets = db_engine.get_datasets()

    files_metadata = []
    dataset_name = datasets.replace(".", "_").replace(" ", "_")

    for added_dataset in added_datasets:
        if dataset_name in added_dataset:
            files_metadata.extend(db_engine.get_files_metadata(added_dataset))

    awaitables = []

    graph_db_type = infrastructure_config.get_config()["graph_engine"]

    graph_client = await get_graph_client(graph_db_type)

    # await initialize_graph(USER_ID, graph_data_model, graph_client)

    for file_metadata in files_metadata[:1]:
        with open(file_metadata["file_path"], "rb") as file:
            try:
                file_type = guess_file_type(file)
                text = extract_text_from_file(file, file_type)

                awaitables.append(process_text(text, file_metadata))
            except FileTypeException:
                logger.warning("File (%s) has an unknown file type. We are skipping it.", file_metadata["id"])

    graphs = await asyncio.gather(*awaitables)

    return graph_client.graph

async def process_text(input_text: str, file_metadata: dict):
    print(f"Processing document ({file_metadata['id']}).")

    graph_client = await get_graph_client(infrastructure_config.get_config()["graph_engine"])

    document_id = await add_document_node(
        graph_client,
        parent_node_id = f"DefaultGraphModel_{USER_ID}",
        document_metadata = file_metadata,
    )

    # await add_keywords_nodes(graph_client, document_id, file_metadata["keywords"].split("|"))

    classified_categories = await get_content_categories(input_text)
    await add_classification_nodes(
        graph_client,
        parent_node_id = document_id,
        categories = classified_categories,
    )

    # print(f"Document ({document_id}) classified.")

    # content_summary = await get_content_summary(input_text)
    # await add_summary_nodes(graph_client, document_id, content_summary)

    # print(f"Document ({document_id}) summarized.")

    # try:
    #     # Classify the content into categories
    #     content_labels = await label_content(
    #         input_text,
    #         infrastructure_config.get_config()["labeling_model"]
    #     )
    #     file_metadata["content_labels"] = content_labels["content_labels"]
    # except Exception as e:
    #     print(e)
    #     raise e


    cognitive_layers = await get_cognitive_layers(input_text, classified_categories)
    cognitive_layers = await add_cognitive_layers(graph_client, document_id, cognitive_layers)

    layer_graphs = await get_layer_graphs(input_text, cognitive_layers)
    # print("Layer graphs are: ", layer_graphs)
    await add_cognitive_layer_graphs(graph_client, document_id, layer_graphs)

    print(f"Document ({document_id}) cognified.")

    # await add_label_nodes(
    #     graph_client,
    #     f"DOCUMENT_{file_metadata['id']}",
    #     { "content_labels": file_metadata["content_labels"] }
    # )

    # await append_to_graph(graph_client, layer_graphs, classified_categories[0])

    # infra_config = infrastructure_config.get_config()

    # db_engine = infra_config["database_engine"]

    # data = [{
    #     "document_id": file_metadata["id"],
    #     "layer_id": base_node_for_graph
    # }]

    # db_engine.load_cognify_data(data)

    # node_descriptions_for_processing_doc = await graph_client.extract_node_description(base_node_for_graph)

    # print("Node descriptions are: ", str(node_descriptions_for_processing_doc))

    # nodes_by_layer_for_processing_doc = await group_nodes_by_layer(node_descriptions_for_processing_doc)
    # unique_layers = nodes_by_layer_for_processing_doc.keys()

    # try:
    #     vector_engine = infrastructure_config.get_config()["vector_engine"]

    #     for layer in unique_layers:
    #         await vector_engine.create_collection(layer)
    # except Exception as e:
    #     print(e)

    # await add_propositions(nodes_by_layer_for_processing_doc)

    # if infrastructure_config.get_config()["connect_documents"] == True:
    #     relevant_documents_to_connect = db_engine.fetch_cognify_data(excluded_document_id=file_metadata["id"])

    #     print("Relevant documents to connect are: ", relevant_documents_to_connect)

    #     list_of_nodes = []

    #     relevant_documents_to_connect.append({
    #         "document_id": file_metadata["id"],
    #         "layer_id": base_node_for_graph,
    #         "created_at": "2024-04-05 16:47:09.651000",
    #         "updated_at": "2024-04-05 16:47:09.651000",
    #     })

    #     for document in relevant_documents_to_connect:
    #         node_descriptions_to_match = await graph_client.extract_node_description(document["layer_id"])
    #         # list_of_nodes.append(node_descriptions_to_match)
    #         list_of_nodes.extend(node_descriptions_to_match)

    #     print("List of nodes are: ", len(list_of_nodes))

    #     nodes_by_layer = await group_nodes_by_layer(list_of_nodes)
    #     print("Nodes by layer are: ", str(nodes_by_layer)[:5000])

    #     results = await resolve_cross_graph_references(nodes_by_layer)
    #     print("Results are: ", str(results)[:3000])

    #     relationships = graph_ready_output(results)

    #     await connect_nodes_in_graph(
    #         graph_client,
    #         relationships,
    #         score_threshold = infrastructure_config.get_config()["intra_layer_score_treshold"]
    #     )

    #     results = await resolve_cross_graph_references(nodes_by_layer)
    

if __name__ == "__main__":

    async def main():


        infrastructure_config.set_config({
            "graph_engine": GraphDBType.NETWORKX
        })
        # print(infrastructure_config.get_config())
        text_1 = """Thomas Mann wrote German novels about horses and nature. Hello novels
        """
        text_2 = """German novels are fun to read and talk about nature"""
        dataset_name = "explanations"
        from cognee.api.v1.add.add import add
        await add(
            [
                text_1,
                text_2

            ],
            dataset_name
        )
        graph = await cognify(datasets=dataset_name)

        # if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
        #
        #     graph_client = await get_graph_client(GraphDBType.NETWORKX)
        #     from cognee.utils import render_graph
        #     graph_url = await render_graph(graph_client.graph, include_nodes=False, include_color=False, include_size=False,include_labels=False)
        #     print(graph_url)


    asyncio.run(main())
