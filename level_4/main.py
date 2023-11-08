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





def execute_cypher_query(query: str):
    graph_ = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="pleaseletmein")
    graph_.query(query)
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


import uuid


def create_base_queries_from_user( user_id: str):
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

    return user_memory_cypher

# Function to append a UUID4 to the variable names to ensure uniqueness
def append_uuid_to_variable_names(variable_mapping):
    unique_variable_mapping = {}
    for original_name in variable_mapping.values():
        unique_name = f"{original_name}_{uuid.uuid4().hex}"
        unique_variable_mapping[original_name] = unique_name
    return unique_variable_mapping

# Update the functions to use the unique variable names
def create_node_variable_mapping(nodes):
    mapping = {}
    for node in nodes:
        variable_name = f"{node['category']}{node['id']}".lower()
        mapping[node['id']] = variable_name
    return mapping


def create_edge_variable_mapping(edges):
    mapping = {}
    for edge in edges:
        # Construct a unique identifier for the edge
        variable_name = f"edge{edge['source']}to{edge['target']}".lower()
        mapping[(edge['source'], edge['target'])] = variable_name
    return mapping


# Update the function to generate Cypher CREATE statements for nodes with unique variable names


def format_dict(d):
    # Initialize an empty list to store formatted items
    formatted_items = []

    # Iterate through all key-value pairs
    for key, value in d.items():
        # Format key-value pairs with a colon and space, and adding quotes for string values
        formatted_item = f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
        formatted_items.append(formatted_item)

    # Join all formatted items with a comma and a space
    formatted_string = ", ".join(formatted_items)

    # Add curly braces to mimic a dictionary
    formatted_string = f"{{{formatted_string}}}"

    return formatted_string
def generate_create_statements_for_nodes_with_uuid(nodes, unique_mapping):
    create_statements = []
    for node in nodes:
        original_variable_name = node_variable_mapping[node['id']]
        unique_variable_name = unique_mapping[original_variable_name]
        node_label = node['category'].capitalize()
        properties = {k: v for k, v in node.items() if k not in ['id', 'category']}
        try:
            properties = format_dict(properties)
        except:
            pass
        create_statements.append(f"CREATE ({unique_variable_name}:{node_label} {properties})")
    return create_statements

# Update the function to generate Cypher CREATE statements for edges with unique variable names
def generate_create_statements_for_edges_with_uuid(edges, unique_mapping):
    create_statements = []
    with_statement = f"WITH {', '.join(unique_mapping.values())}, user, semantic, episodic, buffer"
    create_statements.append(with_statement)

    for edge in edges:
        # print("HERE IS THE EDGE", edge)
        source_variable = unique_mapping[node_variable_mapping[edge['source']]]
        target_variable = unique_mapping[node_variable_mapping[edge['target']]]
        relationship = edge['description'].replace(" ", "_").upper()
        create_statements.append(f"CREATE ({source_variable})-[:{relationship}]->({target_variable})")
    return create_statements


# Update the function to generate Cypher CREATE statements for memory type relationships with unique variable names
def generate_memory_type_relationships_with_uuid_and_time_context(nodes, unique_mapping):
    create_statements = []
    with_statement = f"WITH {', '.join(unique_mapping.values())}, user, semantic, episodic, buffer"
    create_statements.append(with_statement)

    # Loop through each node and create relationships based on memory_type
    for node in nodes:
        original_variable_name = node_variable_mapping[node['id']]
        unique_variable_name = unique_mapping[original_variable_name]
        if node['memory_type'] == 'semantic':
            create_statements.append(f"CREATE (semantic)-[:HAS_KNOWLEDGE]->({unique_variable_name})")
        elif node['memory_type'] == 'episodic':
            create_statements.append(f"CREATE (episodic)-[:HAS_EVENT]->({unique_variable_name})")
            if node['category'] == 'time':
                create_statements.append(f"CREATE (buffer)-[:HAS_TIME_CONTEXT]->({unique_variable_name})")

        # Assuming buffer holds all actions and times
        # if node['category'] in ['action', 'time']:
        create_statements.append(f"CREATE (buffer)-[:CURRENTLY_HOLDING]->({unique_variable_name})")

    return create_statements



# Main execution logic
if __name__ == "__main__":
    user_id = "User1"
    query_input = "I walked in the forest yesterday and added to my list I need to buy some milk in the store"

    # Generate the knowledge graph from the user input
    knowledge_graph = generate_graph(query_input)
    visualize_knowledge_graph(knowledge_graph)
    # out = knowledge_graph.dict()
    # print(out)
    #
    # graph: KnowledgeGraph = generate_graph("I walked in the forest yesterday and added to my list I need to buy some milk in the store")
    # graph_dic = graph.dict()
    #
    # node_variable_mapping = create_node_variable_mapping(graph_dic['nodes'])
    # edge_variable_mapping = create_edge_variable_mapping(graph_dic['edges'])
    # # Create unique variable names for each node
    # unique_node_variable_mapping = append_uuid_to_variable_names(node_variable_mapping)
    # unique_edge_variable_mapping = append_uuid_to_variable_names(edge_variable_mapping)
    # create_nodes_statements = generate_create_statements_for_nodes_with_uuid(graph_dic['nodes'], unique_node_variable_mapping)
    # create_edges_statements = generate_create_statements_for_edges_with_uuid(graph_dic['edges'], unique_node_variable_mapping)
    #
    # memory_type_statements_with_uuid_and_time_context = generate_memory_type_relationships_with_uuid_and_time_context(
    #     graph_dic['nodes'], unique_node_variable_mapping)
    #
    # # # Combine all statements
    # cypher_statements = [create_base_queries_from_user(user_id)] + create_nodes_statements + create_edges_statements + memory_type_statements_with_uuid_and_time_context
    # cypher_statements_joined = "\n".join(cypher_statements)
    #
    # print(cypher_statements_joined)
    #
    # execute_cypher_query(cypher_statements_joined)



    # Translate the KnowledgeGraph into Cypher queries


    # print(cypher_query)
# #
# #     # Execute the Cypher queries to create the graph in Neo4j
#     execute_cypher_query(cypher_query)
# # Refresh the graph schema
# graph.refresh_schema()
#
# # Print the schema to the console
# print(graph.schema)
