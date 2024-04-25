from typing import List
import dspy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from cognee.config import Config
from cognee.shared.data_models import KnowledgeGraph, Node, Edge
from cognee.utils import num_tokens_from_string, trim_text_to_max_tokens

config = Config()
config.load()

# """Instructions:
# You are a top-tier algorithm designed for extracting information from text in structured formats to build a knowledge graph.
# - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
# - **Edges** represent relationships between concepts. They're akin to Wikipedia links.
# Extract as much information as you can from the text and build a detailed knowledge graph.
# If question is provided, make sure that the information to answer the question is present in the graph."""

class GraphFromText(dspy.Signature):
    """Instructions:
    You are a top-tier algorithm designed for extracting information from text in structured formats to build a knowledge graph.
    - **Nodes** represent entities and concepts, akin to Wikipedia nodes.
    - **Edges** represent relationships between entities and concepts, akin to Wikipedia hyperlinks.
    Extract information from the text and build a detailed knowledge graph."""

    text: str = dspy.InputField()
    graph: KnowledgeGraph = dspy.OutputField()


def are_all_nodes_and_edges_valid(graph: KnowledgeGraph) -> bool:
    return all([getattr(node, "entity_type", "").strip() != "" for node in graph.nodes]) and \
        all([getattr(node, "entity_name", "").strip() != "" for node in graph.nodes]) and \
        all([getattr(edge, "relationship_name", "").strip() != "" for edge in graph.edges])

def is_node_connected(node: Node, edges: List[Edge]) -> bool:
    return any([(edge.source_node_id == node.id or edge.target_node_id == node.id) for edge in edges])

def are_all_nodes_connected(graph: KnowledgeGraph) -> bool:
    return all([is_node_connected(node, graph.edges) for node in graph.nodes])


class ExtractKnowledgeGraph(dspy.Module):
    def __init__(self, lm = dspy.OpenAI(model = config.openai_model, api_key = config.openai_key, model_type = "chat", max_tokens = 4096)):
        super().__init__()
        self.lm = lm
        dspy.settings.configure(lm=self.lm)
        self.generate_graph = dspy.TypedChainOfThought(GraphFromText)
        nltk.download("stopwords", quiet = True)

    def forward(self, context: str, question: str):
        context = remove_stop_words(context)
        context = trim_text_to_max_tokens(context, 1500, config.openai_model)
      
        with dspy.context(lm = self.lm):
            graph = self.generate_graph(text = context).graph

            not_valid_nodes_or_edges_message = """
                All nodes must contain "entity_name".
                All edges must contain "relationship_name".
                Please add mandatory fields to nodes and edges."""

            dspy.Suggest(are_all_nodes_and_edges_valid(graph), not_valid_nodes_or_edges_message)

            # not_connected_graph_message = """
            #     Output must be a graph that has all nodes connected to it.
            #     Please find a relation and connect nodes or remove them."""

            # dspy.Suggest(are_all_nodes_connected(graph), not_connected_graph_message)

        return dspy.Prediction(context = context, graph = graph)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)

#
# if __name__ == "__main__":
#     gpt_4_turbo = dspy.OpenAI(model="gpt-4", max_tokens=4000, api_key=config.openai_key, model_type="chat")
#     dspy.settings.configure(lm=gpt_4_turbo)


#     extract_knowledge_graph = ExtractKnowledgeGraph(lm=gpt_4_turbo)
#     # graph_text = extract_knowledge_graph("cognitive_layer", "text")
#     graph = extract_knowledge_graph("analysis_layer", """A large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process. LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
# LLMs are artificial neural networks. The largest and most capable, as of March 2024""", question="What is a large language model?")
#     print("GPT4 History:", gpt_4_turbo.inspect_history(n=1))
#     print(graph)
#
