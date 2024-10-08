from deepeval.dataset import EvaluationDataset
from pydantic import BaseModel


from typing import List, Type
from deepeval.test_case import LLMTestCase
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

async def run_cognify_base_rag():
    from cognee.api.v1.add import add
    from cognee.api.v1.prune import prune
    from cognee.api.v1.cognify.cognify import cognify

    await prune.prune_system()

    await add("data://test_datasets", "initial_test")

    graph = await cognify("initial_test")



    pass


import os
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector import get_vector_engine

async def cognify_search_base_rag(content:str, context:str):
    base_config = get_base_config()

    cognee_directory_path = os.path.abspath(".cognee_system")
    base_config.system_root_directory = cognee_directory_path

    vector_engine = get_vector_engine()

    return_ = await vector_engine.search(collection_name="basic_rag", query_text=content, limit=10)

    print("results", return_)
    return return_

async def cognify_search_graph(content:str, context:str):
    from cognee.api.v1.search import search, SearchType
    params = {'query': 'Donald Trump'}

    results = await search(SearchType.INSIGHTS, params)
    print("results", results)
    return results



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

# # Data preprocessing before setting the dataset test cases
# dataset.test_cases = convert_goldens_to_test_cases(dataset.test_cases)
#
#
# from deepeval.metrics import HallucinationMetric
#
#
# metric = HallucinationMetric()
# dataset.evaluate([metric])


if __name__ == "__main__":

    import asyncio

    async def main():
        # await run_cognify_base_rag()
        # await cognify_search_base_rag("show_all_processes", "context")
        await cognify_search_graph("show_all_processes", "context")
    asyncio.run(main())
    # run_cognify_base_rag_and_search()
    # # Data preprocessing before setting the dataset test cases
    # dataset.test_cases = convert_goldens_to_test_cases(dataset.test_cases)
    # from deepeval.metrics import HallucinationMetric
    # metric = HallucinationMetric()
    # dataset.evaluate([metric])
    pass