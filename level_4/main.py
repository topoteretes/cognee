from enum import Enum

import typer
import os
# import marvin
# from pydantic_settings import BaseSettings
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
# from marvin import ai_classifier
# marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")
DEFAULT_PRESET = "promethai_chat"
preset_options = [DEFAULT_PRESET]
import questionary
PROMETHAI_DIR = os.path.join(os.path.expanduser("~"), ".")



def create_config_dir():
    if not os.path.exists(PROMETHAI_DIR):
        os.makedirs(PROMETHAI_DIR, exist_ok=True)

    folders = ["personas", "humans", "archival", "agents"]
    for folder in folders:
        if not os.path.exists(os.path.join(PROMETHAI_DIR, folder)):
            os.makedirs(os.path.join(PROMETHAI_DIR, folder))



from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Neo4jVector
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

txt_path =  "dune.txt"

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="pleaseletmein")

import openai
import instructor

# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON
instructor.patch()
from pydantic import BaseModel, Field
from typing import List

class Node(BaseModel):
    id: int
    description: str
    category: str
    color: str ="blue"
    memory_type: str

#
# class EntityNode(BaseModel):
#     id: int
#     description: str
#
#
# class TimeContextNode(BaseModel):
#     id: int
#     description: str
#
#
# class ActionNode(BaseModel):
#     id: int
#     description: str


class Edge(BaseModel):
    source: int
    target: int
    description: str
    color: str= "blue"


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)


#

def generate_graph(input) -> KnowledgeGraph:
    return openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": f"""Use the given format to extract information from the following input: {input}. """,

            },
            {   "role":"system", "content": """You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
                - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
                - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
                ## 2. Labeling Nodes
                - **Consistency**: Ensure you use basic or elementary types for node labels.
                  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
                  - Include event, entity, time, or action nodes to the category.
                  - Classify the memory type as episodic or semantic.
                - **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
                ## 3. Handling Numerical Data and Dates
                - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
                - **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
                - **Property Format**: Properties must be in a key-value format.
                - **Quotation Marks**: Never use escaped single or double quotes within property values.
                - **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
                ## 4. Coreference Resolution
                - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
                If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
                always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
                Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
                ## 5. Strict Compliance
                Adhere to the rules strictly. Non-compliance will result in termination."""}
        ],
        response_model=KnowledgeGraph,
    )




# async def memory_route(self, memory_type: str):
#     @ai_classifier
#     class MemoryRoute(Enum):
#         """Represents classifer for type of memories"""
#
#         semantic_memory = "semantic_memory"
#         episodic_memory = "episodic_memory"
#
#
#     namespace = MemoryRoute(str(memory_type))
#
#     return namespace

#
# graph = generate_graph("I went to a walk in the forest in the afternoon and got information from a book.")
# # print("got here")
# #
# print(graph)


def execute_cypher_query(query: str):
    graph.query(query)
    # This is a placeholder for the logic that will execute the Cypher query
    # You would replace this with the actual logic to run the query in your Neo4j database
    print(query)

#Execute Cypher queries to create the user and memory components if they don't exist
#
# graph.query(
#     f"""
#     // Ensure the User node exists
#     MERGE (user:User {{ userId: {user} }})
#
#     // Ensure the SemanticMemory node exists
#     MERGE (semantic:SemanticMemory {{ userId: {user} }})
#     MERGE (user)-[:HAS_SEMANTIC_MEMORY]->(semantic)
#
#     // Ensure the EpisodicMemory node exists
#     MERGE (episodic:EpisodicMemory {{ userId: {user} }})
#     MERGE (user)-[:HAS_EPISODIC_MEMORY]->(episodic)
#
#     // Ensure the Buffer node exists
#     MERGE (buffer:Buffer {{ userId: {user} }})
#     MERGE (user)-[:HAS_BUFFER]->(buffer)
#     """
# )
#
# # Execute Cypher queries to create the cognitive components in the graph
# graph.query(
#     f"""
#     // Parsing the query into components and linking them to the user and memory components
#     MERGE (user:User {{ userId: {user} }})
#     MERGE (semantic:SemanticMemory {{ userId: {user} }})
#     MERGE (episodic:EpisodicMemory {{ userId: {user} }})
#     MERGE (buffer:Buffer {{ userId: {user} }})
#
    # CREATE (action1:Event {{ description: 'take a walk', location: 'forest' }})
    # CREATE (action2:Event {{ description: 'get information', source: 'book' }})
    # CREATE (time:TimeContext {{ description: 'in the afternoon' }})
    #
    # WITH user, semantic, episodic, buffer, action1, action2, time
    # CREATE (knowledge:Knowledge {{ content: 'information from a book' }})
    # CREATE (semantic)-[:HAS_KNOWLEDGE]->(knowledge)
    # CREATE (episodic)-[:HAS_EVENT]->(action1)
    # CREATE (episodic)-[:HAS_EVENT]->(action2)
    # CREATE (episodic)-[:HAS_TIME_CONTEXT]->(time)
    # CREATE (buffer)-[:CURRENTLY_HOLDING]->(action1)
    # CREATE (buffer)-[:CURRENTLY_HOLDING]->(action2)
    # CREATE (buffer)-[:CURRENTLY_HOLDING]->(time)
#     """
# )


def create_cypher_queries_from_graph(graph:str, user_id: str):
    # Create nodes


    # Create the user and memory components if they don't exist
    user_memory_cypher = f"""
    MERGE (user:User {{userId: '{user_id}'}})
    MERGE (semantic:SemanticMemory {{userId: '{user_id}'}})
    MERGE (episodic:EpisodicMemory {{userId: '{user_id}'}})
    MERGE (buffer:Buffer {{userId: '{user_id}'}})
    MERGE (user)-[:HAS_SEMANTIC_MEMORY]->(semantic)
    MERGE (user)-[:HAS_EPISODIC_MEMORY]->(episodic)
    MERGE (user)-[:HAS_BUFFER]->(buffer)
    """

    # Combine all Cypher queries
    combined_cypher_query = f"""
    {user_memory_cypher}
    {graph}
    """

    return combined_cypher_query


from graphviz import Digraph


class Node:
    def __init__(self, id, description, color):
        self.id = id
        self.description = description
        self.color = color

class Edge:
    def __init__(self, source, target, label, color):
        self.source = source
        self.target = target
        self.label = label
        self.color = color
def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.description, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.description, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph.gv", view=True)


# Main execution logic
if __name__ == "__main__":
    user_id = "User1"
    query_input = "I walked in the forest yesterday and added to my list I need to buy some milk in the store"

    # Generate the knowledge graph from the user input
    # knowledge_graph = generate_graph(query_input)
    # out = knowledge_graph.dict()
    # print(out)

    graph: KnowledgeGraph = generate_graph("I walked in the forest yesterday and added to my list I need to buy some milk in the store")
    print(graph.dict())
    visualize_knowledge_graph(graph)



    # Translate the KnowledgeGraph into Cypher queries
    # cypher_query = create_cypher_queries_from_graph(out['graph_query'], user_id)

    # print(cypher_query)
# #
# #     # Execute the Cypher queries to create the graph in Neo4j
#     execute_cypher_query(cypher_query)
# # Refresh the graph schema
# graph.refresh_schema()
#
# # Print the schema to the console
# print(graph.schema)
