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
    connect_nodes_in_graph
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
from cognee.modules.cognify.graph.add_document_node import add_document_node
from cognee.modules.cognify.graph.initialize_graph import initialize_graph
from cognee.infrastructure.files.utils.guess_file_type import guess_file_type
from cognee.infrastructure.files.utils.extract_text_from_file import extract_text_from_file
from cognee.infrastructure import infrastructure_config

config = Config()
config.load()

aclient = instructor.patch(OpenAI())

USER_ID = "default_user"

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


    await initialize_graph(USER_ID, graph_data_model, graph_client)

    print(files_metadata)


    for file_metadata in files_metadata:
        with open(file_metadata["file_path"], "rb") as file:
            file_type = guess_file_type(file)
            text = extract_text_from_file(file, file_type)

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
            infrastructure_config.get_config()["classification_model"]
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
        file_metadata["description"] = content_summary["description"]
    except Exception as e:
        print(e)
        raise e

    try:
        # Classify the content into categories
        content_labels = await label_content(
            input_text,
            "label_content.txt",
            infrastructure_config.get_config()["labeling_model"]
        )
        file_metadata["content_labels"] = content_labels["content_labels"]
    except Exception as e:
        print(e)
        raise e
    graph_client = await get_graph_client(infrastructure_config.get_config()["graph_engine"])


    await add_document_node(graph_client, f"DefaultGraphModel_{USER_ID}", file_metadata)
    print(f"Document ({file_metadata['id']}) categorized: {file_metadata['categories']}")

    cognitive_layers = await content_to_cog_layers(
        classified_categories[0],
        response_model = infrastructure_config.get_config()["congitive_layer_model"]
    )

    cognitive_layers = [layer_subgroup.name for layer_subgroup in cognitive_layers.cognitive_layers]
    import tracemalloc

    tracemalloc.start()


    async def generate_graph_per_layer(text_input: str, layers: List[str], response_model: KnowledgeGraph = KnowledgeGraph):
        generate_graphs_awaitables = [generate_graph(text_input, "generate_graph_prompt.txt", {"layer": layer}, response_model) for layer in
                layers]

        return await asyncio.gather(*generate_graphs_awaitables)

    # Run the async function for each set of cognitive layers
    layer_graphs = await generate_graph_per_layer(input_text, cognitive_layers)

    print("Layer graphs generated %s", layer_graphs)

    print(f"Document ({file_metadata['id']}) layer graphs created")


    base_node_for_graph = await add_classification_nodes(graph_client,f"DOCUMENT_{file_metadata['id']}", classified_categories[0])

    await add_summary_nodes(graph_client,f"DOCUMENT_{file_metadata['id']}", {"summary": file_metadata["summary"]})

    await add_label_nodes(graph_client,f"DOCUMENT_{file_metadata['id']}", {"content_labels": file_metadata["content_labels"]})

    await append_to_graph(graph_client, layer_graphs, classified_categories[0])

    print(f"Document ({file_metadata['id']}) layers connected")

    print("Document categories, summaries and metadata are: ", str(classified_categories))

    print("Document metadata is: ", str(file_metadata))

    print("Base nodes for a graph : ", base_node_for_graph)
    infra_config = infrastructure_config.get_config()
    db_engine = infra_config["database_engine"]
    data = [
        {
            'document_id': file_metadata['id'],
            'layer_id': base_node_for_graph
        },]
    db_engine.load_cognify_data(data)



    # base_node_for_graph ='LLM_CLASSIFICATION_LAYER_Research papers and academic publications_DOCUMENT_062c22df-d99b-599f-90cd-2d325c8bcf69'

    node_descriptions_for_processing_doc = await graph_client.extract_node_description(base_node_for_graph)

    print("Node descriptions are: ", str(node_descriptions_for_processing_doc))

    nodes_by_layer_for_processing_doc = await group_nodes_by_layer(node_descriptions_for_processing_doc)
    unique_layers = nodes_by_layer_for_processing_doc.keys()

    try:
        vector_engine = infrastructure_config.get_config()["vector_engine"]

        for layer in unique_layers:
            await vector_engine.create_collection(layer)
    except Exception as e:
        print(e)

    await add_propositions(nodes_by_layer_for_processing_doc)

    relevant_documents_to_connect = db_engine.fetch_cognify_data(excluded_document_id=file_metadata['id'])
    list_of_nodes =[]
    #
    # relevant_documents_to_connect=[  {'document_id': '6dfe01b6-07d2-5b77-83c8-1d6c11ce2aa7', 'layer_id': 'LLM_CLASSIFICATION_LAYER_Articles, essays, and reports_DOCUMENT_6dfe01b6-07d2-5b77-83c8-1d6c11ce2aa7', 'created_at': '2024-04-05 16:47:09.651000', 'updated_at': '2024-04-05 16:47:09.651000'}]
    relevant_documents_to_connect.append({'document_id': file_metadata['id'], 'layer_id': base_node_for_graph, 'created_at': '2024-04-05 16:47:09.651000', 'updated_at': '2024-04-05 16:47:09.651000'})
    for document in relevant_documents_to_connect:
        node_descriptions_to_match =await graph_client.extract_node_description(document['layer_id'])
        # list_of_nodes.append(node_descriptions_to_match)
        list_of_nodes.extend(node_descriptions_to_match)

    print("List of nodes are: ", len(list_of_nodes))




    nodes_by_layer = await group_nodes_by_layer(list_of_nodes)
    print("Nodes by layer are: ", str(nodes_by_layer)[:5000])
    # nodes_by_layer = {
    #     'uuuOmeKGCeuiOqqemWiOyuaaeaKWKOiiKSGf': [
    #         {'node_id': '1ace9d1a-273e-4466-b9c1-d3889957033d',
    #          'description': 'A language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification',
    #          'layer_uuid': 'd8f31061-0bb8-4312-9f40-d4622c2a89d9',
    #          'layer_decomposition_uuid': 'uuuOmeKGCeuiOqqemWiOyuaaeaKWKOiiKSGf'},
    #         {'node_id': '735305a9-15ed-41de-9fd8-66ce5dc4f111',
    #          'description': 'A computationally intensive process that involves learning statistical relationships from text documents to acquire abilities for natural language processing tasks',
    #          'layer_uuid': 'd8f31061-0bb8-4312-9f40-d4622c2a89d9',
    #          'layer_decomposition_uuid': 'uuuOmeKGCeuiOqqemWiOyuaaeaKWKOiiKSGf'}
    #     ],
    #     'qySSyOCOKuiGKKeyaaaGuKmqWKOiaiCWCGKE': [
    #         {'node_id': '6ecb5771-78fe-4866-a8d7-62a299212b97',
    #          'description': 'A language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification',
    #          'layer_uuid': 'd8f31061-0bb8-4312-9f40-d4622c2a89d9',
    #          'layer_decomposition_uuid': 'qySSyOCOKuiGKKeyaaaGuKmqWKOiaiCWCGKE'},
    #         {'node_id': '5fcdbaad-2de0-4882-b6d2-0846ac74d19f',
    #          'description': 'Relationships learned by language models from text documents',
    #          'layer_uuid': 'd8f31061-0bb8-4312-9f40-d4622c2a89d9',
    #          'layer_decomposition_uuid': 'qySSyOCOKuiGKKeyaaaGuKmqWKOiaiCWCGKE'}
    #     ]
    #     # More layers...
    # }
    results = await resolve_cross_graph_references(nodes_by_layer)
    print("Results are: ", str(results)[:3000])
    relationships = graph_ready_output(results)
    await connect_nodes_in_graph(graph_client, relationships,
                                 score_threshold=infrastructure_config.get_config()["intra_layer_score_treshold"])

    # nodes_by_layer_for_processing_doc = await group_nodes_by_layer(node_descriptions_to_match)

    results = await resolve_cross_graph_references(nodes_by_layer)






    # relationships = graph_ready_output(results)
    # print("RELATIONSHIPS", str(relationships)[:8000])

    # relationships = {
    #     'emmquuaCWiCGOuqiSaOGSiOyWyKuGWeiKquS': [
    #         {
    #             'collection_id': 'emmquuaCWiCGOuqiSaOGSiOyWyKuGWeiKquS',
    #             'searched_node_id': '77a0bbb3-dc13-4fb8-a665-aadcfc04a05f',
    #             'score': 1.0,
    #             'score_metadata': {
    #                 'text': 'A computer that exploits quantum mechanical phenomena to perform computations.'
    #             },
    #             'original_id_for_search': '7393e6e0-6515-46c4-b927-f99b4f635823'
    #         },
    #         {
    #             'collection_id': 'emmquuaCWiCGOuqiSaOGSiOyWyKuGWeiKquS',
    #             'searched_node_id': '77a0bbb3-dc13-4fb8-a665-aadcfc04a05f',
    #             'score': 0.7439015507698059,
    #             'score_metadata': {
    #                 'text': 'The potential ability of quantum computing devices to solve problems that classical computers cannot.'
    #             },
    #             'original_id_for_search': 'b239c21a-0278-4223-8985-20962087c39e'
    #         },
    #         # Additional entries would follow the same structure...
    #         {
    #             'collection_id': 'emmquuaCWiCGOuqiSaOGSiOyWyKuGWeiKquS',
    #             'searched_node_id': 'de774e2a-4d86-4542-8074-c077ad50c1a5',
    #             'score': 1.0,
    #             'score_metadata': {
    #                 'text': 'A computer that exploits quantum mechanical phenomena to perform computations.'
    #             },
    #             'original_id_for_search': '7393e6e0-6515-46c4-b927-f99b4f635823'
    #         }
    #     ]
    # }
    #
    # await connect_nodes_in_graph(graph_client, relationships, score_threshold=infrastructure_config.get_config()["intra_layer_score_treshold"] )
    #
    # print(f"Document ({file_metadata['id']}) processed")
    #
    # return graph



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

        if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:

            graph_client = await get_graph_client(GraphDBType.NETWORKX)
            from cognee.utils import render_graph
            graph_url = await render_graph(graph_client.graph)
            print(graph_url)


    asyncio.run(main())