import dsp
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from cognee.modules.data.extraction.knowledge_graph.extract_knowledge_graph_module import ExtractKnowledgeGraph
from cognee.root_dir import get_absolute_path
from cognee.config import Config
from cognee.shared.data_models import Answer
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.cognify.dataset import HotPotQA

config = Config()
config.load()

def evaluate():
    dataset = HotPotQA(
        train_seed = 1,
        train_size = 16,
        eval_seed = 2023,
        dev_size = 8,
        test_size = 0,
        keep_details = True,
    )

    #Evaluate
    evaluate_examples = [
        Example(
            base = None,
            question = None,
            context = "\r\n".join("".join(sentences) for sentences in example.context["sentences"]),
            answer = example.answer,
        ) for example in dataset.dev
    ]

    devset = [example.with_inputs("context", "question") for example in evaluate_examples]

    evaluate_on_hotpotqa = Evaluate(devset = devset, num_threads = 1, display_progress = True, display_table = 5, max_tokens = 4096)

    gpt4 = dspy.OpenAI(model = config.openai_model, api_key = config.openai_key, model_type = "chat", max_tokens = 4096)
    compiled_extract_knowledge_graph = ExtractKnowledgeGraph(lm = gpt4)
    compiled_extract_knowledge_graph.load(get_absolute_path("./programs/extract_knowledge_graph/extract_knowledge_graph.json"))

    def evaluate_answer(example, graph_prediction, trace = None):
        llm_client = get_llm_client()

        try:
            answer_prediction = llm_client.create_structured_output(
                text_input = example.question,
                system_prompt = f"""Answer the question by looking at the provided knowledge graph.
                Use only the graph to answer the question and be very brief.
                This is the knowledge graph:
                {graph_prediction.graph.model_dump(mode = "json")}""",
                response_model = Answer,
            )
        except:
            return False

        return dsp.answer_match(example.answer, [answer_prediction.answer], frac = 0.8) or \
            dsp.passage_match([example.answer], [answer_prediction.answer])

    gpt4 = dspy.OpenAI(model = config.openai_model, api_key = config.openai_key, model_type = "chat", max_tokens = 4096)
    dspy.settings.configure(lm = gpt4)

    evaluate_on_hotpotqa(compiled_extract_knowledge_graph, metric = evaluate_answer)


if __name__ == "__main__":
    evaluate()
