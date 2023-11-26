# import marvin
# from pydantic_settings import BaseSettings
# from marvin import ai_classifier
# marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")
import logging
import os

print(os.getcwd())

import networkx as nx

from langchain.graphs import Neo4jGraph
import os
from dotenv import load_dotenv

import openai
import instructor
from openai import OpenAI
from openai import AsyncOpenAI
import pickle

from abc import ABC, abstractmethod

# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON

from pydantic import BaseModel, Field
from typing import List
from ...utils import format_dict, append_uuid_to_variable_names, create_edge_variable_mapping, create_node_variable_mapping
DEFAULT_PRESET = "promethai_chat"
preset_options = [DEFAULT_PRESET]
PROMETHAI_DIR = os.path.join(os.path.expanduser("~"), ".")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from ...config import Config

config = Config()
config.load()

print(config.model)
print(config.openai_key)

OPENAI_API_KEY = config.openai_key

aclient = instructor.patch(OpenAI())


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
    out =  aclient.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": f"""Use the given format to extract information from the following input: {input}. """,

            },
            {   "role":"system", "content": """You are a top-tier algorithm
                designed for extracting information in structured formats to build a knowledge graph.
                - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
                - The aim is to achieve simplicity and clarity in the 
                knowledge graph, making it accessible for a vast audience.
                ## 2. Labeling Nodes
                - **Consistency**: Ensure you use basic or elementary types for node labels.
                  - For example, when you identify an entity representing a person,
                   always label it as **"person"**. 
                  Avoid using more specific terms like "mathematician" or "scientist".
                  - Include event, entity, time, or action nodes to the category.
                  - Classify the memory type as episodic or semantic.
                - **Node IDs**: Never utilize integers as node IDs. 
                    Node IDs should be names or human-readable identifiers found in the text.
                ## 3. Handling Numerical Data and Dates
                - Numerical data, like age or other related information, 
                should be incorporated as attributes or properties of the respective nodes.
                - **No Separate Nodes for Dates/Numbers**: 
                Do not create separate nodes for dates or numerical values.
                 Always attach them as attributes or properties of nodes.
                - **Property Format**: Properties must be in a key-value format.
                - **Quotation Marks**: Never use escaped single or double quotes within property values.
                - **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
                ## 4. Coreference Resolution
                - **Maintain Entity Consistency**: 
                When extracting entities, it's vital to ensure consistency.
                If an entity, such as "John Doe", is mentioned multiple times 
                in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
                always use the most complete identifier for that entity throughout the knowledge graph.
                 In this example, use "John Doe" as the entity ID.  
                Remember, the knowledge graph should be coherent and easily understandable,
                 so maintaining consistency in entity references is crucial. 
                ## 5. Strict Compliance
                Adhere to the rules strictly. Non-compliance will result in termination."""}
        ],
        response_model=KnowledgeGraph,
    )
    return out

class AbstractGraphDB(ABC):

    @abstractmethod
    def query(self, query: str, params=None):
        pass

    # @abstractmethod
    # def create_nodes(self, nodes: List[dict]):
    #     pass
    #
    # @abstractmethod
    # def create_edges(self, edges: List[dict]):
    #     pass
    #
    # @abstractmethod
    # def create_memory_type_relationships(self, nodes: List[dict], memory_type: str):
    #     pass


class Neo4jGraphDB(AbstractGraphDB):
    def __init__(self, url, username, password):
        self.graph = Neo4jGraph(url=url, username=username, password=password)
        self.openai_key = config.openai_key



    def query(self, query, params=None):
        return self.graph.query(query, params)



    def create_base_cognitive_architecture(self, user_id: str):
        # Create the user and memory components if they don't exist
        user_memory_cypher = f"""
        MERGE (user:User {{userId: '{user_id}'}})
        MERGE (semantic:SemanticMemory {{description: 'SemanticMemory', userId: '{user_id}' }})
        MERGE (episodic:EpisodicMemory {{description: 'EpisodicMemory' , userId: '{user_id}'}})
        MERGE (buffer:Buffer {{description: 'Buffer' , userId: '{user_id}' }})
        MERGE (user)-[:HAS_SEMANTIC_MEMORY]->(semantic)
        MERGE (user)-[:HAS_EPISODIC_MEMORY]->(episodic)
        MERGE (user)-[:HAS_BUFFER]->(buffer)
        """
        return user_memory_cypher

    def user_query_to_edges_and_nodes(self, input: str) ->KnowledgeGraph:
        return openai.ChatCompletion.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to extract information from the following input: {input}. """,

                },
                {"role": "system", "content": """You are a top-tier algorithm
                    designed for extracting information in structured formats to build a knowledge graph.
                    - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
                    - The aim is to achieve simplicity and clarity in the 
                    knowledge graph, making it accessible for a vast audience.
                    ## 2. Labeling Nodes
                    - **Consistency**: Ensure you use basic or elementary types for node labels.
                      - For example, when you identify an entity representing a person,
                       always label it as **"person"**. 
                      Avoid using more specific terms like "mathematician" or "scientist".
                      - Include event, entity, time, or action nodes to the category.
                      - Classify the memory type as episodic or semantic.
                    - **Node IDs**: Never utilize integers as node IDs. 
                        Node IDs should be names or human-readable identifiers found in the text.
                    ## 3. Handling Numerical Data and Dates
                    - Numerical data, like age or other related information, 
                    should be incorporated as attributes or properties of the respective nodes.
                    - **No Separate Nodes for Dates/Numbers**: 
                    Do not create separate nodes for dates or numerical values.
                     Always attach them as attributes or properties of nodes.
                    - **Property Format**: Properties must be in a key-value format.
                    - **Quotation Marks**: Never use escaped single or double quotes within property values.
                    - **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
                    ## 4. Coreference Resolution
                    - **Maintain Entity Consistency**: 
                    When extracting entities, it's vital to ensure consistency.
                    If an entity, such as "John Doe", is mentioned multiple times 
                    in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
                    always use the most complete identifier for that entity throughout the knowledge graph.
                     In this example, use "John Doe" as the entity ID.  
                    Remember, the knowledge graph should be coherent and easily understandable,
                     so maintaining consistency in entity references is crucial. 
                    ## 5. Strict Compliance
                    Adhere to the rules strictly. Non-compliance will result in termination."""}
            ],
            response_model=KnowledgeGraph,
        )

    def generate_create_statements_for_nodes_with_uuid(self, nodes, unique_mapping, base_node_mapping):
        create_statements = []
        for node in nodes:
            original_variable_name = base_node_mapping[node['id']]
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
    def generate_create_statements_for_edges_with_uuid(self, user_id, edges, unique_mapping, base_node_mapping):
        create_statements = []
        with_statement = f"WITH {', '.join(unique_mapping.values())}, {user_id}, semantic, episodic, buffer"
        create_statements.append(with_statement)

        for edge in edges:
            # print("HERE IS THE EDGE", edge)
            source_variable = unique_mapping[base_node_mapping[edge['source']]]
            target_variable = unique_mapping[base_node_mapping[edge['target']]]
            relationship = edge['description'].replace(" ", "_").upper()
            create_statements.append(f"CREATE ({source_variable})-[:{relationship}]->({target_variable})")
        return create_statements

    def generate_memory_type_relationships_with_uuid_and_time_context(self, user_id, nodes, unique_mapping, base_node_mapping):
        create_statements = []
        with_statement = f"WITH {', '.join(unique_mapping.values())}, {user_id}, semantic, episodic, buffer"
        create_statements.append(with_statement)

        # Loop through each node and create relationships based on memory_type
        for node in nodes:
            original_variable_name = base_node_mapping[node['id']]
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

    async def generate_cypher_query_for_user_prompt_decomposition(self, user_id:str, query:str):

        graph: KnowledgeGraph = generate_graph(query)
        graph_dic = graph.dict()

        node_variable_mapping = create_node_variable_mapping(graph_dic['nodes'])
        edge_variable_mapping = create_edge_variable_mapping(graph_dic['edges'])
        # Create unique variable names for each node
        unique_node_variable_mapping = append_uuid_to_variable_names(node_variable_mapping)
        unique_edge_variable_mapping = append_uuid_to_variable_names(edge_variable_mapping)
        create_nodes_statements = self.generate_create_statements_for_nodes_with_uuid(graph_dic['nodes'], unique_node_variable_mapping, node_variable_mapping)
        create_edges_statements =self.generate_create_statements_for_edges_with_uuid(user_id, graph_dic['edges'], unique_node_variable_mapping, node_variable_mapping)

        memory_type_statements_with_uuid_and_time_context = self.generate_memory_type_relationships_with_uuid_and_time_context(user_id,
            graph_dic['nodes'], unique_node_variable_mapping, node_variable_mapping)

        # # Combine all statements
        cypher_statements = [self.create_base_cognitive_architecture(user_id)] + create_nodes_statements + create_edges_statements + memory_type_statements_with_uuid_and_time_context
        cypher_statements_joined = "\n".join(cypher_statements)
        return cypher_statements_joined


    def update_user_query_for_user_prompt_decomposition(self, user_id, user_query):
        pass


    def delete_all_user_memories(self, user_id):
        try:
            # Check if the user exists
            user_exists = self.graph.query(f"MATCH (user:User {{userId: '{user_id}'}}) RETURN user")
            if not user_exists:
                return f"No user found with ID: {user_id}"

            # Delete all memory nodes and relationships for the given user
            delete_query = f"""
            MATCH (user:User {{userId: '{user_id}'}})-[r]-()
            DELETE r
            WITH user
            MATCH (user)-[:HAS_SEMANTIC_MEMORY]->(semantic)
            MATCH (user)-[:HAS_EPISODIC_MEMORY]->(episodic)
            MATCH (user)-[:HAS_BUFFER]->(buffer)
            DETACH DELETE semantic, episodic, buffer
            """
            self.graph.query(delete_query)
            return f"All memories deleted for user ID: {user_id}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def delete_specific_memory_type(self, user_id, memory_type):
        try:
            # Check if the user exists
            user_exists = self.graph.query(f"MATCH (user:User {{userId: '{user_id}'}}) RETURN user")
            if not user_exists:
                return f"No user found with ID: {user_id}"

            # Validate memory type
            if memory_type not in ['SemanticMemory', 'EpisodicMemory', 'Buffer']:
                return "Invalid memory type. Choose from 'SemanticMemory', 'EpisodicMemory', or 'Buffer'."

            # Delete specific memory type nodes and relationships for the given user
            delete_query = f"""
            MATCH (user:User {{userId: '{user_id}'}})-[:HAS_{memory_type.upper()}]->(memory)
            DETACH DELETE memory
            """
            self.graph.query(delete_query)
            return f"{memory_type} deleted for user ID: {user_id}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
    def retrieve_semantic_memory(self, user_id: str):
        query = f"""
        MATCH (user:User {{userId: '{user_id}' }})-[:HAS_SEMANTIC_MEMORY]->(semantic:SemanticMemory)
        MATCH (semantic)-[:HAS_KNOWLEDGE]->(knowledge)
        RETURN knowledge
        """
        return self.query(query, params={"user_id": user_id})

    def retrieve_episodic_memory(self, user_id: str):
        query = """
        MATCH (user:User {userId: $user_id})-[:HAS_EPISODIC_MEMORY]->(episodic:EpisodicMemory)
        MATCH (episodic)-[:HAS_EVENT]->(event)
        RETURN event
        """
        return self.query(query, params={"user_id": user_id})

    def retrieve_buffer_memory(self, user_id: str):
        query = """
        MATCH (user:User {userId: $user_id})-[:HAS_BUFFER]->(buffer:Buffer)
        MATCH (buffer)-[:CURRENTLY_HOLDING]->(item)
        RETURN item
        """
        return self.query(query, params={"user_id": user_id})
    def generate_graph_semantic_memory_document_summary(self, document_summary : str, unique_graphdb_mapping_values: dict, document_namespace: str):
        """ This function takes a document and generates a document summary in Semantic Memory"""
        create_statements = []
        with_statement = f"WITH {', '.join(unique_graphdb_mapping_values.values())}, user, semantic, episodic, buffer"
        create_statements.append(with_statement)

        # Loop through each node and create relationships based on memory_type

        create_statements.append(f"CREATE (semantic)-[:HAS_KNOWLEDGE]->({unique_graphdb_mapping_values})")


        return create_statements


    def generate_document_summary(self, document_summary : str, unique_graphdb_mapping_values: dict, document_namespace: str):
        """ This function takes a document and generates a document summary in Semantic Memory"""


        # fetch namespace from postgres db
        # fetch 1st and last page from vector store
        # summarize the text, add document type
        # write to postgres
        create_statements = []
        with_statement = f"WITH {', '.join(unique_graphdb_mapping_values.values())}, user, semantic, episodic, buffer"
        create_statements.append(with_statement)

        # Loop through each node and create relationships based on memory_type

        create_statements.append(f"CREATE (semantic)-[:HAS_KNOWLEDGE]->({unique_graphdb_mapping_values})")


        return create_statements

    async def get_document_categories(self, user_id: str):
        """
        Retrieve a list of categories for all documents associated with a given user.

        This function executes a Cypher query in a Neo4j database to fetch the categories
        of all 'Document' nodes that are linked to the 'SemanticMemory' node of the specified user.

        Parameters:
        - session (AsyncSession): The database session for executing the query.
        - user_id (str): The unique identifier of the user.

        Returns:
        - List[str]: A list of document categories associated with the user.

        Raises:
        - Exception: If an error occurs during the database query execution.
        """
        try:
            query = f'''
            MATCH (user:User {{userId: '{user_id}' }})-[:HAS_SEMANTIC_MEMORY]->(semantic:SemanticMemory)-[:HAS_DOCUMENT]->(document:Document)
            RETURN document.documentCategory AS category
            '''
            logging.info(f"Generated Cypher query: {query}")
            return query

        except Exception as e:
            logging.error(f"An error occurred while retrieving document categories: {str(e)}")
            return None

    async def get_document_ids(self, user_id: str, category: str):
        """
        Retrieve a list of document IDs for a specific category associated with a given user.

        This function executes a Cypher query in a Neo4j database to fetch the IDs
        of all 'Document' nodes in a specific category that are linked to the 'SemanticMemory' node of the specified user.

        Parameters:
        - user_id (str): The unique identifier of the user.
        - category (str): The specific document category to filter by.

        Returns:
        - List[str]: A list of document IDs in the specified category associated with the user.

        Raises:
        - Exception: If an error occurs during the database query execution.
        """
        try:
            query = f'''
            MATCH (user:User {{userId: '{user_id}' }})-[:HAS_SEMANTIC_MEMORY]->(semantic:SemanticMemory)-[:HAS_DOCUMENT]->(document:Document {{documentCategory: '{category}'}})
            RETURN document.d_id AS d_id
            '''
            logging.info(f"Generated Cypher query: {query}")
            return query

        except Exception as e:
            logging.error(f"An error occurred while retrieving document IDs: {str(e)}")
            return None

    def create_document_node_cypher(self, document_summary: dict, user_id: str) -> str:
        """
        Generate a Cypher query to create a Document node linked to a SemanticMemory node for a user.

        Parameters:
        - document_summary (dict): A dictionary containing the document's category, title, and summary.
        - user_id (str): The unique identifier for the user.

        Returns:
        - str: A Cypher query string with parameters.

        Raises:
        - ValueError: If any required data is missing or invalid.
        """

        # Validate the input parameters
        if not isinstance(document_summary, dict):
            raise ValueError("The document_summary must be a dictionary.")
        if not all(key in document_summary for key in ['DocumentCategory', 'Title', 'Summary', 'd_id']):
            raise ValueError("The document_summary dictionary is missing required keys.")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("The user_id must be a non-empty string.")

        # Escape single quotes in the document summary data (if not using parameters)
        title = document_summary['Title'].replace("'", "\\'")
        summary = document_summary['Summary'].replace("'", "\\'")
        document_category = document_summary['DocumentCategory'].replace("'", "\\'")
        d_id = document_summary['d_id'].replace("'", "\\'")

        # Generate the Cypher query using parameters
        cypher_query = f'''
        // Ensure the User node exists
        MERGE (user:User {{ userId: '{user_id}' }})
    
        // Ensure the SemanticMemory node exists and is connected to the User
        MERGE (semantic:SemanticMemory {{ userId: '{user_id}' }})
        MERGE (user)-[:HAS_SEMANTIC_MEMORY]->(semantic)
    
        // Create the Document node with its properties
        CREATE (document:Document {{
            title: '{title}',
            summary: '{summary}',
            documentCategory: '{document_category}',
            d_id: '{d_id}'
        }})
    
        // Link the Document node to the SemanticMemory node
        CREATE (semantic)-[:HAS_DOCUMENT]->(document)
        '''
        logging.info(f"Generated Cypher query: {cypher_query}")

        return cypher_query

    def update_document_node_with_namespace(self, user_id: str, vectordb_namespace: str, document_id: str):
        # Generate the Cypher query
        cypher_query = f'''
        MATCH (user:User {{userId: '{user_id}' }})-[:HAS_SEMANTIC_MEMORY]->(semantic:SemanticMemory)-[:HAS_DOCUMENT]->(document:Document {{d_id: '{document_id}'}})
        SET document.vectordbNamespace = '{vectordb_namespace}'
        RETURN document
        '''


        return cypher_query

    def get_namespaces_by_document_category(self, user_id: str, category: str):
        """
        Retrieve a list of Vectordb namespaces for documents of a specified category associated with a given user.

        This function executes a Cypher query in a Neo4j database to fetch the 'vectordbNamespace' of all 'Document' nodes
        that are linked to the 'SemanticMemory' node of the specified user and belong to the specified category.

        Parameters:
        - user_id (str): The unique identifier of the user.
        - category (str): The category to filter the documents by.

        Returns:
        - List[str]: A list of Vectordb namespaces for documents in the specified category.

        Raises:
        - Exception: If an error occurs during the database query execution.
        """
        try:
            query = f'''
            MATCH (user:User {{userId: '{user_id}'}})-[:HAS_SEMANTIC_MEMORY]->(semantic:SemanticMemory)-[:HAS_DOCUMENT]->(document:Document)
            WHERE document.documentCategory = '{category}'
            RETURN document.vectordbNamespace AS namespace
            '''
            result = self.query(query)
            namespaces = [record["namespace"] for record in result]
            return namespaces
        except Exception as e:
            logging.error(f"An error occurred while retrieving namespaces by document category: {str(e)}")
            return None


class NetworkXGraphDB:
    def __init__(self, filename='networkx_graph.pkl'):
        self.filename = filename
        try:
            self.graph = self.load_graph()  # Attempt to load an existing graph
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.graph = nx.Graph()  # Create a new graph if loading failed

    def save_graph(self):
        """ Save the graph to a file using pickle """
        with open(self.filename, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph(self):
        """ Load the graph from a file using pickle """
        with open(self.filename, 'rb') as f:
            return pickle.load(f)

    def create_base_cognitive_architecture(self, user_id: str):
        # Add nodes for user and memory types if they don't exist
        self.graph.add_node(user_id, type='User')
        self.graph.add_node(f"{user_id}_semantic", type='SemanticMemory')
        self.graph.add_node(f"{user_id}_episodic", type='EpisodicMemory')
        self.graph.add_node(f"{user_id}_buffer", type='Buffer')

        # Add edges to connect user to memory types
        self.graph.add_edge(user_id, f"{user_id}_semantic", relation='HAS_SEMANTIC_MEMORY')
        self.graph.add_edge(user_id, f"{user_id}_episodic", relation='HAS_EPISODIC_MEMORY')
        self.graph.add_edge(user_id, f"{user_id}_buffer", relation='HAS_BUFFER')

        self.save_graph()  # Save the graph after modifying it

    def delete_all_user_memories(self, user_id: str):
        # Remove nodes and edges related to the user's memories
        for memory_type in ['semantic', 'episodic', 'buffer']:
            memory_node = f"{user_id}_{memory_type}"
            self.graph.remove_node(memory_node)

        self.save_graph()  # Save the graph after modifying it

    def delete_specific_memory_type(self, user_id: str, memory_type: str):
        # Remove a specific type of memory node and its related edges
        memory_node = f"{user_id}_{memory_type.lower()}"
        if memory_node in self.graph:
            self.graph.remove_node(memory_node)

        self.save_graph()  # Save the graph after modifying it

    def retrieve_semantic_memory(self, user_id: str):
        return [n for n in self.graph.neighbors(f"{user_id}_semantic")]

    def retrieve_episodic_memory(self, user_id: str):
        return [n for n in self.graph.neighbors(f"{user_id}_episodic")]

    def retrieve_buffer_memory(self, user_id: str):
        return [n for n in self.graph.neighbors(f"{user_id}_buffer")]

    def generate_graph_semantic_memory_document_summary(self, document_summary, unique_graphdb_mapping_values, document_namespace, user_id):
        for node, attributes in unique_graphdb_mapping_values.items():
            self.graph.add_node(node, **attributes)
            self.graph.add_edge(f"{user_id}_semantic", node, relation='HAS_KNOWLEDGE')
        self.save_graph()

    def generate_document_summary(self, document_summary, unique_graphdb_mapping_values, document_namespace, user_id):
        self.generate_graph_semantic_memory_document_summary(document_summary, unique_graphdb_mapping_values, document_namespace, user_id)

    async def get_document_categories(self, user_id):
        return [self.graph.nodes[n]['category'] for n in self.graph.neighbors(f"{user_id}_semantic") if 'category' in self.graph.nodes[n]]

    async def get_document_ids(self, user_id, category):
        return [n for n in self.graph.neighbors(f"{user_id}_semantic") if self.graph.nodes[n].get('category') == category]

    def create_document_node(self, document_summary, user_id):
        d_id = document_summary['d_id']
        self.graph.add_node(d_id, **document_summary)
        self.graph.add_edge(f"{user_id}_semantic", d_id, relation='HAS_DOCUMENT')
        self.save_graph()

    def update_document_node_with_namespace(self, user_id, vectordb_namespace, document_id):
        if self.graph.has_node(document_id):
            self.graph.nodes[document_id]['vectordbNamespace'] = vectordb_namespace
        self.save_graph()

    def get_namespaces_by_document_category(self, user_id, category):
        return [self.graph.nodes[n].get('vectordbNamespace') for n in self.graph.neighbors(f"{user_id}_semantic") if self.graph.nodes[n].get('category') == category]

class GraphDBFactory:
    def create_graph_db(self, db_type, **kwargs):
        if db_type == 'neo4j':
            return Neo4jGraphDB(**kwargs)
        elif db_type == 'networkx':
            return NetworkXGraphDB(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")




