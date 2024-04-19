from typing import List, Optional
import dspy
from cognee.config import Config
from cognee.shared.data_models import KnowledgeGraph, Node, Edge
import dotenv
dotenv.load_dotenv()
from cognee.utils import num_tokens_from_string, trim_text_to_max_tokens

from dsp.modules.cache_utils import CacheMemory
print(CacheMemory)



config = Config()
config.load()

class GraphTextFromText(dspy.Signature):
    """Instructions:
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
    - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
    - **Edges** represent relationships between concepts. They're akin to Wikipedia links.
    - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.

    YOU ARE ONLY EXTRACTING DATA FOR COGNITIVE LAYER defined by 'cognitive_layer' input.

    1. Labeling Nodes
    - **Consistency**: Ensure you use basic or elementary types for node labels.
      - "node.entity_type" is mandatory and should be "Person", "Country", "City", "DateTime", "Animal", "Organization", "Venue", "Event" and so on.
      - For example, when you identify an entity representing a person, always label it as **"Person"**. Avoid using more specific terms like "Mathematician" or "Scientist".
      - Never utilize integers for "node.id".
      - "node.id" should be names or human-readable identifiers found in the text.
    2. Handling Numerical Data
      - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
      - **Property Format**: Properties must be in a key-value format.
      - **Quotation Marks**: Never use escaped single or double quotes within property values.
      - **Naming Convention**: Use snake_case for property keys, e.g., `birth_date`.
    3. Coreference Resolution
      - **Maintain Entity Consistency**:
      - When extracting entities, it's vital to ensure consistency.
      - If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
        always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
      - The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
    4. Strict Compliance
       - Adhere to the rules strictly. Non-compliance will result in termination"""

    text: str = dspy.InputField()
    cognitive_layer: Optional[str] = dspy.InputField(desc = "Name of the cognitive layer for which the graph should be created.")
    graph: KnowledgeGraph = dspy.OutputField(desc = "Knowledge graph generated from text, based on the provided cognitive layer.")

# class GraphFromText(dspy.Signature):
#     """Instructions:
#     Take "graph_text" input and verify that it is a valid knowledge graph.
#     Correct mistakes that lead to incorrect knowledge graph."""
#
#     graph_text: str = dspy.InputField()
#     graph: KnowledgeGraph = dspy.OutputField(desc = "Knowledge graph generated from text, based on the provided cognitive layer.")


def are_all_nodes_and_edges_valid(graph: KnowledgeGraph) -> bool:
    return all([getattr(node, "entity_type", "").strip() != "" for node in graph.nodes]) and \
        all([getattr(node, "entity_name", "").strip() != "" for node in graph.nodes]) and \
        all([getattr(edge, "relationship_name", "").strip() != "" for edge in graph.edges])

def is_node_connected(node: Node, edges: List[Edge]) -> bool:
    return any([(edge.source_node_id == node.id or edge.target_node_id == node.id) for edge in edges])

def are_all_nodes_connected(graph: KnowledgeGraph) -> bool:
    return all([is_node_connected(node, graph.edges) for node in graph.nodes])


class ExtractKnowledgeGraph(dspy.Module):
    def __init__(self, lm = dspy.OpenAI(
        model = "gpt-4-turbo",
        max_tokens = 4000
    )):
        super().__init__()
        self.lm = lm
        dspy.settings.configure(lm=self.lm)
        # self.generate_graph_text = dspy.TypedChainOfThought(GraphTextFromText)
        self.generate_graph = dspy.TypedChainOfThought(GraphTextFromText)

    def forward(self, layer: str, text: str):
        text = trim_text_to_max_tokens(text, 2500, config.openai_model)
        with dspy.settings.context(lm=self.lm):

            # graph_text = self.generate_graph_text(text = text, cognitive_layer = layer).graph_text
            graph = self.generate_graph(text = text, cognitive_layer = layer).graph

            not_valid_nodes_or_edges_message = """
                All nodes must contain 'entity_name'.
                All edges must contain 'relationship_name'.
                Please add mandatory fields to nodes and edges."""

            dspy.Suggest(are_all_nodes_and_edges_valid(graph), not_valid_nodes_or_edges_message)

            # not_connected_graph_message = """
            #     Output must be a graph that has all nodes connected to it.
            #     Please find a relation and connect nodes or remove them."""

            # dspy.Suggest(are_all_nodes_connected(graph), not_connected_graph_message)

        return graph


if __name__ == "__main__":
    gpt_4_turbo = dspy.OpenAI(model="gpt-4", max_tokens=6000, api_key=config.openai_key, model_type="chat")
    dspy.settings.configure(lm=gpt_4_turbo)
    extract_knowledge_graph = ExtractKnowledgeGraph(lm=gpt_4_turbo)
    # graph_text = extract_knowledge_graph("cognitive_layer", "text")
    graph = extract_knowledge_graph("analysis_layer", """A large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process. LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
LLMs are artificial neural networks. The largest and most capable, as of March 2024""")
    # print(graph_text)
    print(graph)