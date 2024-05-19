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
    file_path="synthetic_data/20240519_185842.json",
    input_key_name="query",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
    retrieval_context_key_name="retrieval_context",
)



import logging
from typing import List, Dict
from cognee.infrastructure import infrastructure_config

logger = logging.getLogger(__name__)

def AnswerModel(BaseModel):
    response:str
def get_answer_base(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = "Answer the following question: and use the context"

    return  llm_client.create_structured_output(content, system_prompt, response_model)
def get_answer(content: str, model: Type[BaseModel]= AnswerModel):

    try:
        return (get_answer_base(
            content,
            model
        ))
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info = True)
        raise error




def convert_goldens_to_test_cases(goldens: List[Golden]) -> List[LLMTestCase]:
    test_cases = []
    for golden in goldens:
        test_case = LLMTestCase(
            input=golden.input,
            # Generate actual output using the 'input' and 'additional_metadata'
            actual_output= get_answer(golden.input),
            expected_output=golden.expected_output,
            context=golden.context,
        )
        test_cases.append(test_case)
    return test_cases

# Data preprocessing before setting the dataset test cases
dataset.test_cases = convert_goldens_to_test_cases(dataset.goldens)


from deepeval.metrics import HallucinationMetric


metric = HallucinationMetric()
dataset.evaluate([metric])