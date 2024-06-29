import asyncio
import logging
import os
from typing import List, Type

from pydantic import BaseModel
from dotenv import load_dotenv

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, BaseMetric
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.api.v1.add import add
from cognee.api.v1.prune import prune
from cognee.api.v1.cognify.cognify import cognify
from cognee.api.v1.search.search import search

load_dotenv()

logger = logging.getLogger(__name__)


class AnswerModel(BaseModel):
    response: str


async def run_cognify_task(dataset: str = 'test_datasets', dataset_name: str = 'initial_test') -> str:
    """
    Runs the cognify task asynchronously.

    Args:
        dataset (str): Dataset path.
        dataset_name (str): Name of the dataset.

    Returns:
        str: Resulting graph information.
    """
    await prune.prune_system()
    await add(f"data://{dataset}", dataset_name)
    graph = await cognify(dataset_name)
    return graph


async def cognify_search_base_rag(content: str) -> List[str]:
    """
    Searches using base RAG for the given content.

    Args:
        content (str): Content to search for.

    Returns:
        List[str]: List of search results.
    """
    base_config = get_base_config()
    cognee_directory_path = os.path.abspath(".cognee_system")
    base_config.system_root_directory = cognee_directory_path
    vector_engine = get_vector_engine()
    return await vector_engine.search(collection_name="basic_rag", query_text=content, limit=10)


async def cognify_search_graph(content: str, search_type: str = 'SIMILARITY') -> List[str]:
    """
    Searches using graph search for the given content.

    Args:
        content (str): Content to search for.
        search_type (str, optional): Type of search. Defaults to 'SIMILARITY'.

    Returns:
        List[str]: List of search results.
    """
    params = {'query': content}
    return await search(search_type, params)


def get_answer_base(content: str, context: str, response_model: Type[BaseModel]) -> BaseModel:
    """
    Retrieves an answer using the LLM client with given content and context.

    Args:
        content (str): Content for which answer is needed.
        context (str): Context information.
        response_model (Type[BaseModel]): Response model to use.

    Returns:
        BaseModel: Instance of the response model with the answer.
    """
    llm_client = get_llm_client()
    system_prompt = "THIS IS YOUR CONTEXT: " + str(context)
    return llm_client.create_structured_output(content, system_prompt, response_model)


async def get_answer(content: str, context, model: Type[BaseModel] = AnswerModel) -> BaseModel:
    """
    Retrieves an answer asynchronously with error handling.

    Args:
        content (str): Content for which answer is needed.
        context: Context information.
        model (Type[BaseModel], optional): Response model to use. Defaults to AnswerModel.

    Returns:
        BaseModel: Instance of the response model with the answer.
    """
    try:
        # answer = get_answer_base(content, context, model)
        return get_answer_base(content, context, model)
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info=True)
        raise error


async def convert_goldens_to_test_cases(test_cases_raw: List[LLMTestCase], context_type: str = None) -> List[
    LLMTestCase]:
    """
    Converts raw test cases into LLMTestCase objects with actual outputs generated.

    Args:
        test_cases_raw (List[LLMTestCase]): List of raw test cases.
        context_type (str, optional): Type of context to use. Defaults to None.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects with actual outputs populated.
    """
    test_cases = []
    for case in test_cases_raw:
        if context_type == "naive_rag":
            context = await cognify_search_base_rag(case.input)
        elif context_type == "graph":
            context = await cognify_search_graph(case.input)
        else:
            context = case.context

        actual_output = str((await get_answer(case.input, context)).response)

        test_case = LLMTestCase(
            input=case.input,
            actual_output=actual_output,
            expected_output=case.expected_output,
            context=case.context,
            retrieval_context=["retrieval_context"],
        )
        test_cases.append(test_case)

    return test_cases

async def load_dataset(file_path: str, input_key_name: str, actual_output_key_name: str,
                       expected_output_key_name: str, context_key_name: str) -> EvaluationDataset:
    """
    Loads a dataset from a JSON file into an EvaluationDataset object.

    Args:
        file_path (str): Path to the JSON file.
        input_key_name (str): Key name for input data.
        actual_output_key_name (str): Key name for actual output data.
        expected_output_key_name (str): Key name for expected output data.
        context_key_name (str): Key name for context data.

    Returns:
        EvaluationDataset: Loaded EvaluationDataset object.
    """
    dataset = EvaluationDataset()
    dataset.add_test_cases_from_json_file(
        file_path=file_path,
        input_key_name=input_key_name,
        actual_output_key_name=actual_output_key_name,
        expected_output_key_name=expected_output_key_name,
        context_key_name=context_key_name
    )
    return dataset

from typing import Type

async def evaluate_dataset(dataset: EvaluationDataset, context_type: str = None, metric_type: Type[BaseMetric] = HallucinationMetric):
    """
    Evaluates a dataset using specified context type and metric type.

    Args:
        dataset (EvaluationDataset): Dataset to evaluate.
        context_type (str, optional): Type of context to use. Defaults to None.
        metric_type (Type[BaseMetric], optional): Type of metric to use for evaluation. Defaults to HallucinationMetric.
    """
    dataset.test_cases = await convert_goldens_to_test_cases(dataset.test_cases, context_type=context_type)
    metric = metric_type()
    dataset.evaluate([metric])

async def main(file_path: str = "./synthetic_data/20240519_185842.json",
               input_key_name: str = "input",
               actual_output_key_name: str = "actual_output",
               expected_output_key_name: str = "expected_output",
               context_key_name: str = "context",
               context_type: str = "naive_rag"):
    """
    Main function to orchestrate the evaluation process.

    Args:
        file_path (str, optional): Path to the JSON file. Defaults to "./synthetic_data/20240519_185842.json".
        input_key_name (str, optional): Key name for input data. Defaults to "input".
        actual_output_key_name (str, optional): Key name for actual output data. Defaults to "actual_output".
        expected_output_key_name (str, optional): Key name for expected output data. Defaults to "expected_output".
        context_key_name (str, optional): Key name for context data. Defaults to "context".
        context_type (str, optional): Type of context to use. Defaults to "naive_rag".
    """
    dataset = await load_dataset(file_path, input_key_name, actual_output_key_name, expected_output_key_name, context_key_name)
    await evaluate_dataset(dataset, context_type)
# async def main():
#     """
#     Main function to orchestrate the evaluation process.
#     """
#     dataset = EvaluationDataset()
#     dataset.add_test_cases_from_json_file(
#         file_path="./synthetic_data/20240519_185842.json",
#         input_key_name="input",
#         actual_output_key_name="actual_output",
#         expected_output_key_name="expected_output",
#         context_key_name="context"
#     )
#
#     dataset.test_cases = await convert_goldens_to_test_cases(dataset.test_cases, context_type="naive_rag")
#
#     metric = HallucinationMetric()
#     dataset.evaluate([metric])


if __name__ == "__main__":
    asyncio.run(main())
