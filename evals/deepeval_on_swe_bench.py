from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.base_config import get_base_config
import os
import logging
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from typing import List, Dict, Type
from swebench.harness.utils import load_swebench_dataset
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from deepeval.synthesizer import Synthesizer


# DeepEval dataset for reference
# synthesizer = Synthesizer()
# synthesizer.generate_goldens_from_docs(
#     document_paths=['/app/.data/short_stories/soldiers_home.pdf'],
#     include_expected_output=True
# )

def convert_swe_to_deepeval(swe_dataset: List[Dict]):
    deepeval_dataset = EvaluationDataset()
    for datum in swe_dataset:
        input = datum["problem_statement"]
        expected_output = datum["patch"]
        context = [datum["text"]]
        # retrieval_context = datum.get(retrieval_context_key_name)

        deepeval_dataset.add_test_case(
            LLMTestCase(
                input=input,
                actual_output=None,
                expected_output=expected_output,
                context=context,
                # retrieval_context=retrieval_context,
            )
        )
    return deepeval_dataset


swe_dataset = load_swebench_dataset(
    'princeton-nlp/SWE-bench_bm25_13K', split='test')
deepeval_dataset = convert_swe_to_deepeval(swe_dataset)


logger = logging.getLogger(__name__)


class AnswerModel(BaseModel):
    response: str


def get_answer_base(content: str, context: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = "THIS IS YOUR CONTEXT:" + str(context)

    return llm_client.create_structured_output(content, system_prompt, response_model)


def get_answer(content: str, context, model: Type[BaseModel] = AnswerModel):

    try:
        return (get_answer_base(
            content,
            context,
            model
        ))
    except Exception as error:
        logger.error(
            "Error extracting cognitive layers from content: %s", error, exc_info=True)
        raise error


async def run_cognify_base_rag():
    from cognee.api.v1.add import add
    from cognee.api.v1.prune import prune
    from cognee.api.v1.cognify.cognify import cognify

    await prune.prune_system()

    await add("data://test_datasets", "initial_test")

    graph = await cognify("initial_test")
    pass


async def cognify_search_base_rag(content: str, context: str):
    base_config = get_base_config()

    cognee_directory_path = os.path.abspath(".cognee_system")
    base_config.system_root_directory = cognee_directory_path

    vector_engine = get_vector_engine()

    return_ = await vector_engine.search(collection_name="basic_rag", query_text=content, limit=10)

    print("results", return_)
    return return_


async def cognify_search_graph(content: str, context: str):
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
            actual_output=str(get_answer(
                case.input, case.context).model_dump()['response']),
            expected_output=case.expected_output,
            context=case.context,
            retrieval_context=["retrieval_context"],
        )
        test_cases.append(test_case)
    return test_cases


def convert_swe_to_deepeval_testcases(swe_dataset: List[Dict]):
    deepeval_dataset = EvaluationDataset()
    for datum in swe_dataset[:4]:
        input = datum["problem_statement"]
        expected_output = datum["patch"]
        context = [datum["text"]]
        # retrieval_context = datum.get(retrieval_context_key_name)
        # tools_called = datum.get(tools_called_key_name)
        # expected_tools = json_obj.get(expected_tools_key_name)

        deepeval_dataset.add_test_case(
            LLMTestCase(
                input=input,
                actual_output=str(get_answer(
                    input, context).model_dump()['response']),
                expected_output=expected_output,
                context=context,
                # retrieval_context=retrieval_context,
                # tools_called=tools_called,
                # expected_tools=expected_tools,
            )
        )
    return deepeval_dataset


swe_dataset = load_swebench_dataset(
    'princeton-nlp/SWE-bench_bm25_13K', split='test')
test_dataset = convert_swe_to_deepeval_testcases(swe_dataset)

if __name__ == "__main__":

    import asyncio

    async def main():
        # await run_cognify_base_rag()
        # await cognify_search_base_rag("show_all_processes", "context")
        await cognify_search_graph("show_all_processes", "context")
    asyncio.run(main())
    # run_cognify_base_rag_and_search()
    # # Data preprocessing before setting the dataset test cases
    swe_dataset = load_swebench_dataset(
        'princeton-nlp/SWE-bench_bm25_13K', split='test')
    test_dataset = convert_swe_to_deepeval_testcases(swe_dataset)
    from deepeval.metrics import HallucinationMetric
    metric = HallucinationMetric()
    evalresult = test_dataset.evaluate([metric])
    pass
