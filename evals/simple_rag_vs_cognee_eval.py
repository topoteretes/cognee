from deepeval.dataset import EvaluationDataset
from pydantic import BaseModel


from typing import List, Type
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
import dotenv
dotenv.load_dotenv()

from cognee.infrastructure.llm.get_llm_client import get_llm_client

dataset = EvaluationDataset()
dataset.add_test_cases_from_json_file(
    # file_path is the absolute path to you .json file
    file_path="./synthetic_data/20240519_185842.json",
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context"
)

print(dataset)
# from deepeval.synthesizer import Synthesizer
#
# synthesizer = Synthesizer(model="gpt-3.5-turbo")
#
# dataset = EvaluationDataset()
# dataset.generate_goldens_from_docs(
#     synthesizer=synthesizer,
#     document_paths=['natural_language_processing.txt', 'soldiers_home.pdf', 'trump.txt'],
#     max_goldens_per_document=10,
#     num_evolutions=5,
#     enable_breadth_evolve=True,
# )


print(dataset.goldens)
print(dataset)




import logging
from typing import List, Dict
from cognee.infrastructure import infrastructure_config

logger = logging.getLogger(__name__)

class AnswerModel(BaseModel):
    response:str
def get_answer_base(content: str, context:str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = "THIS IS YOUR CONTEXT:" + str(context)

    return  llm_client.create_structured_output(content, system_prompt, response_model)
def get_answer(content: str,context, model: Type[BaseModel]= AnswerModel):

    try:
        return (get_answer_base(
            content,
            context,
            model
        ))
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info = True)
        raise error

def run_cognify_base_rag_and_search():
    pass


def run_cognify_and_search():
    pass



def convert_goldens_to_test_cases(test_cases_raw: List[LLMTestCase]) -> List[LLMTestCase]:
    test_cases = []
    for case in test_cases_raw:
        test_case = LLMTestCase(
            input=case.input,
            # Generate actual output using the 'input' and 'additional_metadata'
            actual_output= str(get_answer(case.input, case.context).model_dump()['response']),
            expected_output=case.expected_output,
            context=case.context,
            retrieval_context=["retrieval_context"],
        )
        test_cases.append(test_case)
    return test_cases

# Data preprocessing before setting the dataset test cases
dataset.test_cases = convert_goldens_to_test_cases(dataset.test_cases)


from deepeval.metrics import HallucinationMetric


metric = HallucinationMetric()
dataset.evaluate([metric])