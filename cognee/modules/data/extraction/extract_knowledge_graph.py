from typing import List, Optional
import dspy
from cognee.config import Config
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.llm.get_llm_client import get_llm_client
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
    Be brief and clear in your response.
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
    - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
    - **Edges** represent relationships between concepts. They're akin to Wikipedia links.
        """

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
    def __init__(self, lm = dspy.OpenAI(model="gpt-4", api_key=config.openai_key, model_type="chat")):
        super().__init__()
        self.lm = lm
        dspy.settings.configure(lm=self.lm)
        # self.generate_graph_text = dspy.TypedChainOfThought(GraphTextFromText)
        self.generate_graph = dspy.TypedChainOfThought(GraphTextFromText)

    def forward(self, question:str, context:str, layer:str = None):
        # context = " ".join(context['sentences'])
        # print("type of context", type(context))

        print("context", str(type(context)))


        # context = trim_text_to_max_tokens(context, 2300, config.openai_model)

        with dspy.settings.context(lm=self.lm):

            graph = self.generate_graph(text = str(context), cognitive_layer = str(layer)).graph

            from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client

            graph_client = get_llm_client()
            from cognee.shared.data_models import Answer

            generate_answer =  graph_client.create_structured_output(text_input = question, system_prompt = f"Create a knowledge graph from the given text. Use only context. be very brief. this is the context {graph}", response_model = Answer)
            generate_answer = generate_answer

            print("generate_answer SSS ", generate_answer)


            not_valid_nodes_or_edges_message = """
                All nodes must contain 'entity_name'.
                All edges must contain 'relationship_name'.
                Please add mandatory fields to nodes and edges."""

            dspy.Suggest(are_all_nodes_and_edges_valid(graph), not_valid_nodes_or_edges_message)

            # not_connected_graph_message = """
            #     Output must be a graph that has all nodes connected to it.
            #     Please find a relation and connect nodes or remove them."""

            # dspy.Suggest(are_all_nodes_connected(graph), not_connected_graph_message)

        pred_object =dspy.Prediction(context=context['sentences'], answer=generate_answer)

        print("pred_object", pred_object)


        return generate_answer



if __name__ == "__main__":
    gpt_4_turbo = dspy.OpenAI(model="gpt-4", max_tokens=4000, api_key=config.openai_key, model_type="chat")
    dspy.settings.configure(lm=gpt_4_turbo)


    extract_knowledge_graph = ExtractKnowledgeGraph(lm=gpt_4_turbo)
    # graph_text = extract_knowledge_graph("cognitive_layer", "text")
    graph = extract_knowledge_graph("analysis_layer", """A large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process. LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
LLMs are artificial neural networks. The largest and most capable, as of March 2024""", question="What is a large language model?")
    print("bbb", gpt_4_turbo.inspect_history(n=1))
    # print(graph_text)
    print(graph)