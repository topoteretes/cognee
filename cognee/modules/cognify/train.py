import dsp
import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
# from .dspy import GraphTextFromText

from cognee.modules.data.extraction.extract_knowledge_graph import ExtractKnowledgeGraph


def train():
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url = "http://20.102.90.50:2017/wiki17_abstracts")

    dspy.configure(rm = colbertv2_wiki17_abstracts)

    def evaluate_answer(example, prediction, frac = 0.8):
        print("example", example)
        print("prediction", prediction)
        return dsp.answer_match(example.answer, [prediction.answer], frac = 0.8) or \
            dsp.passage_match([example.answer], [prediction.answer])

    teleprompter = BootstrapFewShot(metric = evaluate_answer)

    dataset = HotPotQA(
        train_seed = 1,
        train_size = 30,
        eval_seed = 2023,
        dev_size = 50,
        test_size = 0,
        keep_details = True
    )


    for x in dataset.train:
        x.__setattr__("layer", "semantic_layer")
        print(x.question, x.context, x.answer,x.layer)

    trainset = [x.with_inputs("question", "context", "layer") for x in dataset.train][:30]

    # print("Training on HotPotQA dataset...", trainset)

    compiled_rag = teleprompter.compile(ExtractKnowledgeGraph(), trainset = trainset)
    # compiled_rag(context = context[0], layer = "cognitive_layer",)

    return compiled_rag


if __name__ == "__main__":
    train()
