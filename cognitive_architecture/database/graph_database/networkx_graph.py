import pickle

import networkx as nx


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
