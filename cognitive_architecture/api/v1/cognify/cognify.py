import asyncio
import logging
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
import json
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
import re

from dotenv import load_dotenv
import os

from cognitive_architecture.infrastructure.databases.vector.qdrant.adapter import CollectionConfig
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.modules.cognify.graph.add_classification_nodes import add_classification_nodes
from cognitive_architecture.modules.cognify.graph.add_node_connections import add_node_connection, graph_ready_output, \
    connect_nodes_in_graph, extract_node_descriptions
from cognitive_architecture.modules.cognify.graph.add_propositions import append_to_graph
from cognitive_architecture.modules.cognify.llm.add_node_connection_embeddings import process_items
from cognitive_architecture.modules.cognify.vector.batch_search import adapted_qdrant_batch_search
from cognitive_architecture.modules.cognify.vector.load_propositions import add_propositions
from cognitive_architecture.utils import render_graph

# Load environment variables from .env file
load_dotenv()
import instructor
from openai import OpenAI


aclient = instructor.patch(OpenAI())

from typing import Optional, List, Type
from pydantic import BaseModel, Field

from cognitive_architecture.modules.cognify.llm.classify_content import classify_into_categories
from cognitive_architecture.modules.cognify.llm.content_to_cog_layers import content_to_cog_layers
from cognitive_architecture.modules.cognify.llm.content_to_propositions import generate_graph
from cognitive_architecture.shared.data_models import DefaultContentPrediction,  KnowledgeGraph, DefaultCognitiveLayer
from cognitive_architecture.modules.cognify.graph.create import create_semantic_graph
from typing import  Optional, Any
from pydantic import BaseModel
from cognitive_architecture.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognitive_architecture.shared.data_models import GraphDBType, DefaultGraphModel, Document, DocumentType, Category, Relationship, UserProperties, UserLocation
from qdrant_client import models
from cognitive_architecture.infrastructure.databases.vector.get_vector_database import get_vector_database


async def cognify(input_text:str):
    """This function is responsible for the cognitive processing of the content."""
    # Load the content from the text file

    input_article_one= input_text
    # Classify the content into categories

    required_layers_one = await classify_into_categories(input_article_one, "classify_content.txt",
                                                         DefaultContentPrediction)

    def transform_dict(original):
        # Extract the first subclass from the list (assuming there could be more)
        subclass_enum = original['label']['subclass'][0]

        # The data type is derived from 'type' and converted to lowercase
        data_type = original['label']['type'].lower()

        # The context name is the name of the Enum member (e.g., 'NEWS_STORIES')
        context_name = subclass_enum.name.replace('_', ' ').title()

        # The layer name is the value of the Enum member (e.g., 'News stories and blog posts')
        layer_name = subclass_enum.value

        # Construct the new dictionary
        new_dict = {
            'data_type': data_type,
            'context_name': data_type.upper(),  # llm context classification
            'layer_name': layer_name  # llm layer classification
        }

        return new_dict

    # Transform the original dictionary
    transformed_dict_1 = transform_dict(required_layers_one.dict())
    cognitive_layers_one = await content_to_cog_layers("generate_cog_layers.txt", transformed_dict_1,
                                                       response_model=DefaultCognitiveLayer)
    cognitive_layers_one = [layer_subgroup.name for layer_subgroup in cognitive_layers_one.cognitive_layers]

    async def generate_graphs_for_all_layers(text_input: str, layers: List[str], response_model: Type[BaseModel]):
        tasks = [generate_graph(text_input, "generate_graph_prompt.txt", {'layer': layer}, response_model) for layer in
                 layers]
        return await asyncio.gather(*tasks)

    # Execute the async function and print results for each set of layers
    async def async_graph_per_layer(text_input: str, cognitive_layers: List[str]):
        graphs = await generate_graphs_for_all_layers(text_input, cognitive_layers, KnowledgeGraph)
        # for layer, graph in zip(cognitive_layers, graphs):
        #     print(f"{layer}: {graph}")
        return graphs

    # Run the async function for each set of cognitive layers
    layer_1_graph = await async_graph_per_layer(input_article_one, cognitive_layers_one)
    # print(layer_1_graph)
    #
    #
    #
    graph_client = get_graph_client(GraphDBType.NETWORKX)
    #
    # ADD SUMMARY
    # ADD CATEGORIES

    # Define a GraphModel instance with example data
    graph_model_instance = DefaultGraphModel(
        id="user123",
        documents=[
            Document(
                doc_id="doc1",
                title="Document 1",
                summary="Summary of Document 1",
                content_id="content_id_for_doc1",
                doc_type=DocumentType(type_id="PDF", description="Portable Document Format"),
                categories=[
                    Category(category_id="finance", name="Finance",
                             default_relationship=Relationship(type="belongs_to")),
                    Category(category_id="tech", name="Technology",
                             default_relationship=Relationship(type="belongs_to"))
                ],
                default_relationship=Relationship(type='has_document')
            ),
            Document(
                doc_id="doc2",
                title="Document 2",
                summary="Summary of Document 2",
                content_id="content_id_for_doc2",
                doc_type=DocumentType(type_id="TXT", description="Text File"),
                categories=[
                    Category(category_id="health", name="Health", default_relationship=Relationship(type="belongs_to")),
                    Category(category_id="wellness", name="Wellness",
                             default_relationship=Relationship(type="belongs_to"))
                ],
                default_relationship=Relationship(type='has_document')
            )
        ],
        user_properties=UserProperties(
            custom_properties={"age": "30"},
            location=UserLocation(location_id="ny", description="New York",
                                  default_relationship=Relationship(type='located_in'))
        ),
        default_fields={
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    G = await create_semantic_graph(graph_model_instance, graph_client)

    await add_classification_nodes(graph_client, 'Document:doc1', transformed_dict_1)

    F, unique_layer_uuids = await append_to_graph(layer_1_graph, required_layers_one, graph_client)

    # # Extract the node descriptions
    await graph_client.load_graph_from_file()
    graph = graph_client.graph
    node_descriptions = await extract_node_descriptions(graph.nodes(data=True))
    unique_layer_uuids = set(node['layer_decomposition_uuid'] for node in node_descriptions)
    # #
    db = get_vector_database()
    #
    #
    collection_config = CollectionConfig(
        vector_config={
            'content': models.VectorParams(
                distance=models.Distance.COSINE,
                size=3072
            )
        },
        # Set other configs as needed
    )

    from qdrant_client import  QdrantClient
    try:
        for layer in unique_layer_uuids:
            await db.create_collection(layer,collection_config)
    except:
        pass

    # qdrant = QdrantClient(
    #     url=os.getenv('QDRANT_URL'),
    #     api_key=os.getenv('QDRANT_API_KEY'))
    #
    # collections_response = qdrant.http.collections_api.get_collections()
    # collections = collections_response.result.collections
    # print(collections)


    # print(node_descriptions)
    #
    await add_propositions(node_descriptions)

    from cognitive_architecture.infrastructure.databases.vector.qdrant.adapter import AsyncQdrantClient

    grouped_data = await add_node_connection(graph_client, db, node_descriptions)

    # print("we are here, grouped_data", grouped_data)

    llm_client = get_llm_client()



    relationship_dict = await process_items(grouped_data, unique_layer_uuids, llm_client)


    # print("we are here", relationship_dict[0])

    results = await adapted_qdrant_batch_search(relationship_dict, db)
    # print(results)
    relationship_d = graph_ready_output(results)
    # print(relationship_d)

    CONNECTED_GRAPH = connect_nodes_in_graph(F, relationship_d)

    out = await render_graph(CONNECTED_GRAPH.graph, graph_type='networkx')
    print(out)
    return CONNECTED_GRAPH

    #
    # grouped_data = {}
    #
    # # Iterate through each dictionary in the list
    # for item in node_descriptions:
    #     # Get the layer_decomposition_uuid of the current dictionary
    #     uuid = item['layer_decomposition_uuid']
    #
    #     # Check if this uuid is already a key in the grouped_data dictionary
    #     if uuid not in grouped_data:
    #         # If not, initialize a new list for this uuid
    #         grouped_data[uuid] = []
    #
    #     # Append the current dictionary to the list corresponding to its uuid
    #     grouped_data[uuid].append(item)







if __name__ == "__main__":
    asyncio.run(cognify("""In the nicest possible way, Britons have always been a bit silly about animals. “Keeping pets, for the English, is not so much a leisure activity as it is an entire way of life,” wrote the anthropologist Kate Fox in Watching the English, nearly 20 years ago. Our dogs, in particular, have been an acceptable outlet for emotions and impulses we otherwise keep strictly controlled – our latent desire to be demonstratively affectionate, to be silly and chat to strangers. If this seems like an exaggeration, consider the different reactions you’d get if you struck up a conversation with someone in a park with a dog, versus someone on the train.
Indeed, British society has been set up to accommodate these four-legged ambassadors. In the UK – unlike Australia, say, or New Zealand – dogs are not just permitted on public transport but often openly encouraged. Many pubs and shops display waggish signs, reading, “Dogs welcome, people tolerated”, and have treat jars on their counters. The other day, as I was waiting outside a cafe with a friend’s dog, the barista urged me to bring her inside.
For years, Britons’ non-partisan passion for animals has been consistent amid dwindling common ground. But lately, rather than bringing out the best in us, our relationship with dogs is increasingly revealing us at our worst – and our supposed “best friends” are paying the price.
As with so many latent traits in the national psyche, it all came unleashed with the pandemic, when many people thought they might as well make the most of all that time at home and in local parks with a dog. Between 2019 and 2022, the number of pet dogs in the UK rose from about nine million to 13 million. But there’s long been a seasonal surge around this time of year, substantial enough for the Dogs Trust charity to coin its famous slogan back in 1978: “A dog is for life, not just for Christmas.”
"""))