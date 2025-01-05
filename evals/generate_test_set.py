from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
import dotenv
from deepeval.test_case import LLMTestCase

# import pytest
# from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

dotenv.load_dotenv()

# synthesizer = Synthesizer()
# synthesizer.generate_goldens_from_docs(
#     document_paths=['natural_language_processing.txt', 'soldiers_home.pdf', 'trump.txt'],
#     max_goldens_per_document=5,
#     num_evolutions=5,
#     include_expected_output=True,
#     enable_breadth_evolve=True,
# )
#
# synthesizer.save_as(
#     file_type='json', # or 'csv'
#     directory="./synthetic_data"
# )


dataset = EvaluationDataset()
dataset.generate_goldens_from_docs(
    document_paths=["natural_language_processing.txt", "soldiers_home.pdf", "trump.txt"],
    max_goldens_per_document=10,
    num_evolutions=5,
    enable_breadth_evolve=True,
)


print(dataset.goldens)
print(dataset)


answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

# from deepeval import evaluate


# evaluate(dataset, [answer_relevancy_metric])
