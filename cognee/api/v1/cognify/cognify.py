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

    print("WE ARE HERE")

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

    # base_node_for_graph = 'LLM_CLASSIFICATION_LAYER_Research papers and academic publications_DOCUMENT_062c22df-d99b-599f-90cd-2d325c8bcf69'
    #
    # base_node_for_graph ='LLM_CLASSIFICATION_LAYER_Research papers and academic publications_DOCUMENT_062c22df-d99b-599f-90cd-2d325c8bcf69'

    node_descriptions = await graph_client.extract_node_description(base_node_for_graph)


    #
    nodes_by_layer = await group_nodes_by_layer(node_descriptions)
    unique_layers = nodes_by_layer.keys()

    try:
        vector_engine = infrastructure_config.get_config()["vector_engine"]

        for layer in unique_layers:
            await vector_engine.create_collection(layer)
    except Exception as e:
        print(e)

    await add_propositions(nodes_by_layer)
    results = await resolve_cross_graph_references(nodes_by_layer)
    relationships = graph_ready_output(results)
    print("RELATIONSHIPS", str(relationships)[:8000])

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

    graph = await connect_nodes_in_graph(graph_client, relationships, score_threshold=infrastructure_config.get_config()["intra_layer_score_treshold"] )

    print(f"Document ({file_metadata['id']}) processed")

    return graph



if __name__ == "__main__":

    async def main():


        infrastructure_config.set_config({
            "graph_engine": GraphDBType.NETWORKX
        })
        # print(infrastructure_config.get_config())
        text_1 = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
        At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
        Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
        The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
        Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
        In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
        """
        dataset_name = "explanations"
        # from cognee.api.v1.add.add import add
        # await add(
        #     [
        #         text_1
        #
        #     ],
        #     dataset_name
        # )
        graph = await cognify(datasets=dataset_name)

        if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:

            graph_client = await get_graph_client(GraphDBType.NETWORKX)
            from cognee.utils import render_graph
            graph_url = await render_graph(graph_client.graph)
            print(graph_url)


    asyncio.run(main())