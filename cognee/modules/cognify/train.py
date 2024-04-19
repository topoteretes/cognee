import dsp
import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from cognee.modules.data.extraction.extract_knowledge_graph import ExtractKnowledgeGraph

def train():
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url = "http://20.102.90.50:2017/wiki17_abstracts")
    dspy.configure(rm = colbertv2_wiki17_abstracts)

    def evaluate_answer(example, prediction, frac = 0.8):
        return dsp.answer_match(example.answer, [prediction.answer], frac = frac) or \
            dsp.passage_match([example.answer], [prediction.answer])

    teleprompter = BootstrapFewShot(metric = evaluate_answer)

    dataset = HotPotQA(
        train_seed = 1,
        train_size = 30,
        eval_seed = 2023,
        dev_size = 50,
        test_size = 0
    )

    trainset = [x.with_inputs("question") for x in dataset.train][:30]

    compiled_rag = teleprompter.compile(ExtractKnowledgeGraph(), trainset = trainset)

    return compiled_rag
