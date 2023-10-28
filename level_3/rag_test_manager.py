import argparse
import json
import random
import itertools


import logging
import string
from enum import Enum


import openai
from deepeval.metrics.overall_score import OverallScoreMetric
from deepeval.run_test import run_test
from deepeval.test_case import LLMTestCase
from marvin import ai_classifier
from sqlalchemy.future import select

logging.basicConfig(level=logging.INFO)
import marvin
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from database.database import (
    engine,
)  # Ensure you have database engine defined somewhere
from models.user import User
from models.memory import MemoryModel
from models.sessions import Session
from models.testset import TestSet
from models.testoutput import TestOutput
from models.metadatas import MetaDatas
from models.operation import Operation

load_dotenv()
import ast
import tracemalloc
from database.database_crud import session_scope, add_entity

tracemalloc.start()

import os
from dotenv import load_dotenv
import uuid

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")
from vectordb.basevectordb import BaseMemory
from vectorstore_manager import Memory
import asyncio

from database.database_crud import session_scope
from database.database import AsyncSessionLocal

openai.api_key = os.getenv("OPENAI_API_KEY", "")


async def retrieve_latest_test_case(session, user_id, memory_id):
    try:
        # Use await with session.execute() and row.fetchone() or row.all() for async query execution
        result = await session.execute(
            session.query(TestSet.attributes_list)
            .filter_by(user_id=user_id, memory_id=memory_id)
            .order_by(TestSet.created_at)
            .first()
        )
        return (
            result.scalar_one_or_none()
        )  # scalar_one_or_none() is a non-blocking call
    except Exception as e:
        logging.error(
            f"An error occurred while retrieving the latest test case: {str(e)}"
        )
        return None
def get_document_names(doc_input):
    """
    Get a list of document names.

    This function takes doc_input, which can be a folder path, a single document file path, or a document name as a string.
    It returns a list of document names based on the doc_input.

    Args:
        doc_input (str): The doc_input can be a folder path, a single document file path, or a document name as a string.

    Returns:
        list: A list of document names.

    Example usage:
        - Folder path: get_document_names(".data")
        - Single document file path: get_document_names(".data/example.pdf")
        - Document name provided as a string: get_document_names("example.docx")
    """
    if os.path.isdir(doc_input):
        # doc_input is a folder
        folder_path = doc_input
        document_names = []
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                document_names.append(filename)
        return document_names
    elif os.path.isfile(doc_input):
        # doc_input is a single document file
        return [os.path.basename(doc_input)]
    elif isinstance(doc_input, str):
        # doc_input is a document name provided as a string
        return [doc_input]
    else:
        # doc_input is not valid
        return []

async def add_entity(session, entity):
    async with session_scope(session) as s:  # Use your async session_scope
        s.add(entity)  # No need to commit; session_scope takes care of it

        return "Successfully added entity"


async def retrieve_job_by_id(session, user_id, job_id):
    try:
        result = await session.execute(
            session.query(Session.id)
            .filter_by(user_id=user_id, id=job_id)
            .order_by(Session.created_at)
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logging.error(f"An error occurred while retrieving the job: {str(e)}")
        return None


async def fetch_job_id(session, user_id=None, memory_id=None, job_id=None):
    try:
        result = await session.execute(
            session.query(Session.id)
            .filter_by(user_id=user_id, id=job_id)
            .order_by(Session.created_at)
            .first()
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None


async def fetch_test_set_id(session, user_id, content):
    try:
        # Await the execution of the query and fetching of the result
        result = await session.execute(select(TestSet.id)
            .filter_by(user_id=user_id, content=content)
            .order_by(TestSet.created_at)

        )
        return (
            result.scalar_one_or_none()
        )  # scalar_one_or_none() is a non-blocking call
    except Exception as e:
        logging.error(f"An error occurred while retrieving the test set: {str(e)}")
        return None


# Adding "embeddings" to the parameter variants function


def generate_param_variants(
    base_params=None, increments=None, ranges=None, included_params=None
):
    """Generate parameter variants for testing.

    Args:
        base_params (dict): Base parameter values.
        increments (dict): Increment values for each parameter variant.
        ranges (dict): Range (number of variants) to generate for each parameter.
        included_params (list, optional): Parameters to include in the combinations.
                                          If None, all parameters are included.

    Returns:
        list: A list of dictionaries containing parameter variants.
    """

    # Default values
    defaults = {
        "chunk_size": 250,
        "chunk_overlap": 20,
        "similarity_score": 0.5,
        "metadata_variation": 0,
        "search_type": "hybrid",
        "embeddings": "openai",  # Default value added for 'embeddings'
    }

    # Update defaults with provided base parameters
    params = {**defaults, **(base_params or {})}

    default_increments = {
        "chunk_size": 150,
        "chunk_overlap": 10,
        "similarity_score": 0.1,
        "metadata_variation": 1,
    }

    # Update default increments with provided increments
    increments = {**default_increments, **(increments or {})}

    # Default ranges
    default_ranges = {
        "chunk_size": 2,
        "chunk_overlap": 2,
        "similarity_score": 2,
        "metadata_variation": 2,
    }

    # Update default ranges with provided ranges
    ranges = {**default_ranges, **(ranges or {})}

    # Generate parameter variant ranges
    param_ranges = {
        key: [
            params[key] + i * increments.get(key, 1) for i in range(ranges.get(key, 1))
        ]
        for key in [
            "chunk_size",
            "chunk_overlap",
            "similarity_score",
            "metadata_variation",
        ]
    }

    # Add search_type and embeddings with possible values
    param_ranges["search_type"] = [
        "text",
        "hybrid",
        "bm25",
        # "generate",
        # "generate_grouped",
    ]
    param_ranges["embeddings"] = [
        "openai",
        "cohere",
        "huggingface",
    ]  # Added 'embeddings' values

    # Filter param_ranges based on included_params
    if included_params is not None:
        param_ranges = {
            key: val for key, val in param_ranges.items() if key in included_params
        }

    # Generate all combinations of parameter variants
    keys = param_ranges.keys()
    values = param_ranges.values()
    param_variants = [
        dict(zip(keys, combination)) for combination in itertools.product(*values)
    ]

    logging.info("Param combinations for testing", str(param_variants))

    return param_variants



async def generate_chatgpt_output(query: str, context: str = None, api_key=None, model_name="gpt-3.5-turbo"):
    """
    Generate a response from the OpenAI ChatGPT model.

    Args:
        query (str): The user's query or message.
        context (str, optional): Additional context for the conversation. Defaults to an empty string.
        api_key (str, optional): Your OpenAI API key. If not provided, the globally configured API key will be used.
        model_name (str, optional): The name of the ChatGPT model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        str: The response generated by the ChatGPT model.

    Raises:
        Exception: If an error occurs during the API call, an error message is returned for the caller to handle.
    """
    if not context:
        context = ""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": context},
        {"role": "user", "content": query},
    ]

    try:
        openai.api_key = api_key if api_key else openai.api_key  # Use the provided API key or the one set globally

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
        )

        llm_output = response.choices[0].message.content
        return llm_output
    except Exception as e:
        return f"An error occurred: {e}"  # Return the error message for the caller to handle



async def eval_test(
    query=None,
    output=None,
    expected_output=None,
    context=None,
    synthetic_test_set=False,
):
    result_output = await generate_chatgpt_output(query, context)

    if synthetic_test_set:
        test_case = synthetic_test_set
    else:
        test_case = LLMTestCase(
            input=query,
            actual_output=result_output,
            expected_output=expected_output,
            context=context,
        )
    metric = OverallScoreMetric()

    # If you want to run the test
    test_result = run_test(test_case, metrics=[metric], raise_error=False)

    def test_result_to_dict(test_result):
        return {
            "success": test_result.success,
            "score": test_result.score,
            "metric_name": test_result.metric_name,
            "query": test_result.query,
            "output": test_result.output,
            "expected_output": test_result.expected_output,
            "metadata": test_result.metadata,
            "context": test_result.context,
        }

    test_result_dict = []
    for test in test_result:
        test_result_it = test_result_to_dict(test)
        test_result_dict.append(test_result_it)
    return test_result_dict
    # You can also inspect the test result class
    # print(test_result)


def count_files_in_data_folder(data_folder_path=".data"):
    try:
        # Get the list of files in the specified folder
        files = os.listdir(data_folder_path)

        # Count the number of files
        file_count = len(files)

        return file_count
    except FileNotFoundError:
        return 0  # Return 0 if the folder does not exist
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return -1  # Return -1 to indicate an error
# def data_format_route(data_string: str):
#     @ai_classifier
#     class FormatRoute(Enum):
#         """Represents classifier for the data format"""
#
#         PDF = "PDF"
#         UNSTRUCTURED_WEB = "UNSTRUCTURED_WEB"
#         GITHUB = "GITHUB"
#         TEXT = "TEXT"
#         CSV = "CSV"
#         WIKIPEDIA = "WIKIPEDIA"
#
#     return FormatRoute(data_string).name


def data_format_route(data_string: str):
    class FormatRoute(Enum):
        """Represents classifier for the data format"""

        PDF = "PDF"
        UNSTRUCTURED_WEB = "UNSTRUCTURED_WEB"
        GITHUB = "GITHUB"
        TEXT = "TEXT"
        CSV = "CSV"
        WIKIPEDIA = "WIKIPEDIA"

    # Convert the input string to lowercase for case-insensitive matching
    data_string = data_string.lower()

    # Mapping of keywords to categories
    keyword_mapping = {
        "pdf": FormatRoute.PDF,
        "web": FormatRoute.UNSTRUCTURED_WEB,
        "github": FormatRoute.GITHUB,
        "text": FormatRoute.TEXT,
        "csv": FormatRoute.CSV,
        "wikipedia": FormatRoute.WIKIPEDIA
    }

    # Try to match keywords in the data string
    for keyword, category in keyword_mapping.items():
        if keyword in data_string:
            return category.name

    # Return a default category if no match is found
    return FormatRoute.PDF.name

def data_location_route(data_string: str):
    class LocationRoute(Enum):
        """Represents classifier for the data location, if it is device, or database connection string or URL"""

        DEVICE = "DEVICE"
        URL = "URL"
        DATABASE = "DATABASE"

    # Convert the input string to lowercase for case-insensitive matching
    data_string = data_string.lower()

    # Check for specific patterns in the data string
    if data_string.startswith(".data") or "data" in data_string:
        return LocationRoute.DEVICE.name
    elif data_string.startswith("http://") or data_string.startswith("https://"):
        return LocationRoute.URL.name
    elif "postgres" in data_string or "mysql" in data_string:
        return LocationRoute.DATABASE.name

    # Return a default category if no match is found
    return "Unknown"

def dynamic_test_manager(context=None):
    from deepeval.dataset import create_evaluation_query_answer_pairs

    # fetch random chunks from the document
    # feed them to the evaluation pipeline
    dataset = create_evaluation_query_answer_pairs(
        openai_api_key=os.environ.get("OPENAI_API_KEY"), context=context, n=10
    )

    return dataset


def generate_letter_uuid(length=8):
    """Generate a random string of uppercase letters with the specified length."""
    letters = string.ascii_uppercase  # A-Z
    return "".join(random.choice(letters) for _ in range(length))


async def start_test(
    data,
    test_set=None,
    user_id=None,
    params=None,
    metadata=None,
    generate_test_set=False,
    retriever_type: str = None,
):
    """retriever_type = "llm_context, single_document_context, multi_document_context, "cognitive_architecture""" ""

    async with session_scope(session=AsyncSessionLocal()) as session:
        job_id = ""
        job_id = await fetch_job_id(session, user_id=user_id, job_id=job_id)
        test_set_id = await fetch_test_set_id(session, user_id=user_id, content=str(test_set))
        memory = await Memory.create_memory(
            user_id, session, namespace="SEMANTICMEMORY"
        )
        await memory.add_memory_instance("ExampleMemory")
        existing_user = await Memory.check_existing_user(user_id, session)

        if test_set_id is None:
            test_set_id = str(uuid.uuid4())
            await add_entity(
                session, TestSet(id=test_set_id, user_id=user_id, content=str(test_set))
            )

        if params is None:
            data_format = data_format_route(
                data
            )  # Assume data_format_route is predefined
            logging.info("Data format is %s", data_format)
            data_location = data_location_route(data)
            logging.info(
                "Data location is %s", data_location
            )  # Assume data_location_route is predefined
            test_params = generate_param_variants(included_params=["chunk_size"])
        if params:
            data_format = data_format_route(
                data
            )  # Assume data_format_route is predefined
            logging.info("Data format is %s", data_format)
            data_location = data_location_route(data)
            logging.info(
                "Data location is %s", data_location
            )
            test_params = generate_param_variants(included_params=params)

        logging.info("Here are the test params %s", str(test_params))

        loader_settings = {
            "format": f"{data_format}",
            "source": f"{data_location}",
            "path": data,
        }
        if job_id is None:
            job_id = str(uuid.uuid4())

            await add_entity(
                session,
                Operation(
                    id=job_id,
                    user_id=user_id,
                    operation_params=str(test_params),
                    number_of_files=count_files_in_data_folder(),
                    operation_type=retriever_type,
                    test_set_id=test_set_id,
                ),
            )
            doc_names = get_document_names(data)
            for doc in doc_names:

                await add_entity(
                    session,
                    Docs(
                        id=str(uuid.uuid4()),
                        operation_id=job_id,
                        doc_name = doc
                    )
                )

        async def run_test(
            test, loader_settings, metadata, test_id=None, retriever_type=False
        ):
            if test_id is None:
                test_id = str(generate_letter_uuid()) + "_" + "SEMANTICMEMORY"
            await memory.manage_memory_attributes(existing_user)
            test_class = test_id + "_class"
            await memory.add_dynamic_memory_class(test_id.lower(), test_id)
            dynamic_memory_class = getattr(memory, test_class.lower(), None)
            methods_to_add = ["add_memories", "fetch_memories", "delete_memories"]

            if dynamic_memory_class is not None:
                for method_name in methods_to_add:
                    await memory.add_method_to_class(dynamic_memory_class, method_name)
                    print(f"Memory method {method_name} has been added")
            else:
                print(f"No attribute named {test_class.lower()} in memory.")

            print(f"Trying to access: {test_class.lower()}")
            print("Available memory classes:", await memory.list_memory_classes())
            if test:
                loader_settings.update(test)
            # Check if the search_type is 'none'
            if loader_settings.get('search_type') == 'none':
                # Change it to 'hybrid'
                loader_settings['search_type'] = 'hybrid'

            test_class = test_id + "_class"
            dynamic_memory_class = getattr(memory, test_class.lower(), None)

            async def run_load_test_element(
                loader_settings=loader_settings,
                metadata=metadata,
                test_id=test_id,
                test_set=test_set,
            ):
                print(f"Trying to access: {test_class.lower()}")
                await memory.dynamic_method_call(
                    dynamic_memory_class,
                    "add_memories",
                    observation="Observation loaded",
                    params=metadata,
                    loader_settings=loader_settings,
                )
                return "Loaded test element"

            async def run_search_element(test_item, test_id, search_type="text"):
                retrieve_action = await memory.dynamic_method_call(
                    dynamic_memory_class,
                    "fetch_memories",
                    observation=str(test_item["question"]), search_type=loader_settings.get('search_type'),
                )
                print(
                    "Here is the test result",
                    str(retrieve_action),
                )
                if loader_settings.get('search_type') == 'bm25':
                    return retrieve_action["data"]["Get"][test_id]
                else:
                    return retrieve_action["data"]["Get"][test_id][0]["text"]

            async def run_eval(test_item, search_result):
                test_eval = await eval_test(
                    query=test_item["question"],
                    expected_output=test_item["answer"],
                    context=str(search_result),
                )
                return test_eval

            async def run_generate_test_set(test_id):
                test_class = test_id + "_class"
                # await memory.add_dynamic_memory_class(test_id.lower(), test_id)
                dynamic_memory_class = getattr(memory, test_class.lower(), None)
                print(dynamic_memory_class)
                retrieve_action = await memory.dynamic_method_call(
                    dynamic_memory_class,
                    "fetch_memories",
                    observation="Generate a short summary of this document",
                    search_type="generative",
                )
                return dynamic_test_manager(retrieve_action)

            test_eval_pipeline = []



            if retriever_type == "llm_context":
                for test_qa in test_set:
                    context = ""
                    logging.info("Loading and evaluating test set for LLM context")
                    test_result = await run_eval(test_qa, context)
                    test_eval_pipeline.append(test_result)
            elif retriever_type == "single_document_context":
                if test_set:
                    logging.info(
                        "Loading and evaluating test set for a single document context"
                    )
                    await run_load_test_element(
                        loader_settings, metadata, test_id, test_set
                    )
                    for test_qa in test_set:
                        result = await run_search_element(test_qa, test_id)
                        test_result = await run_eval(test_qa, result)
                        test_result.append(test)
                        test_eval_pipeline.append(test_result)
                    await memory.dynamic_method_call(
                        dynamic_memory_class, "delete_memories", namespace=test_id
                    )
                else:
                    pass
            if generate_test_set is True:
                synthetic_test_set = run_generate_test_set(test_id)
            else:
                pass

            return test_id, test_eval_pipeline

        results = []

        logging.info("Validating the retriever type")

        logging.info("Retriever type: %s", retriever_type)

        if retriever_type == "llm_context":
            logging.info("Retriever type: llm_context")
            test_id, result = await run_test(
                test=None,
                loader_settings=loader_settings,
                metadata=metadata,
                retriever_type=retriever_type,
            )  # No params for this case
            results.append([result, "No params"])

        elif retriever_type == "single_document_context":
            logging.info("Retriever type: single document context")
            for param in test_params:
                logging.info("Running for chunk size %s", param["chunk_size"])
                test_id, result = await run_test(
                    param, loader_settings, metadata, retriever_type=retriever_type
                )  # Add the params to the result
                # result.append(param)
                results.append(result)

        for b in results:
            logging.info("Loading  %s", str(b))
            for result, chunk in b:
                logging.info("Loading  %s", str(result))
                await add_entity(
                    session,
                    TestOutput(
                        id=test_id,
                        test_set_id=test_set_id,
                        operation_id=job_id,
                        set_id=str(uuid.uuid4()),
                        user_id=user_id,
                        test_results=result["success"],
                        test_score=str(result["score"]),
                        test_metric_name=result["metric_name"],
                        test_query=result["query"],
                        test_output=result["output"],
                        test_expected_output=str(["expected_output"]),
                        test_context=result["context"][0],
                        test_params=str(chunk),  # Add params to the database table
                    ),
                )

        return results


async def main():
    metadata = {
        "version": "1.0",
        "agreement_id": "AG123456",
        "privacy_policy": "https://example.com/privacy",
        "terms_of_service": "https://example.com/terms",
        "format": "json",
        "schema_version": "1.1",
        "checksum": "a1b2c3d4e5f6",
        "owner": "John Doe",
        "license": "MIT",
        "validity_start": "2023-08-01",
        "validity_end": "2024-07-31",
    }

    test_set = [
        {
            "question": "Who is the main character in 'The Call of the Wild'?",
            "answer": "Buck",
        },
        {"question": "Who wrote 'The Call of the Wild'?", "answer": "Jack London"},
        {
            "question": "Where does Buck live at the start of the book?",
            "answer": "In the Santa Clara Valley, at Judge Miller’s place.",
        },
        {
            "question": "Why is Buck kidnapped?",
            "answer": "He is kidnapped to be sold as a sled dog in the Yukon during the Klondike Gold Rush.",
        },
        {
            "question": "How does Buck become the leader of the sled dog team?",
            "answer": "Buck becomes the leader after defeating the original leader, Spitz, in a fight.",
        },
    ]
    # "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    # http://public-library.uk/ebooks/59/83.pdf
    # result = await start_test(
    #     ".data/3ZCCCW.pdf",
    #     test_set=test_set,
    #     user_id="677",
    #     params=["chunk_size", "search_type"],
    #     metadata=metadata,
    #     retriever_type="single_document_context",
    # )

    parser = argparse.ArgumentParser(description="Run tests against a document.")
    parser.add_argument("--file", required=True, help="URL or location of the document to test.")
    parser.add_argument("--test_set", required=True, help="Path to JSON file containing the test set.")
    parser.add_argument("--user_id", required=True, help="User ID.")
    parser.add_argument("--params", help="Additional parameters in JSON format.")
    parser.add_argument("--metadata", required=True, help="Path to JSON file containing metadata.")
    # parser.add_argument("--generate_test_set", required=False, help="Make a test set.")
    parser.add_argument("--retriever_type", required=False, help="Do a test only within the existing LLM context")
    args = parser.parse_args()

    try:
        with open(args.test_set, "r") as file:
            test_set = json.load(file)
            if not isinstance(test_set, list):  # Expecting a list
                raise TypeError("Parsed test_set JSON is not a list.")
    except Exception as e:
        print(f"Error loading test_set: {str(e)}")
        return

    try:
        with open(args.metadata, "r") as file:
            metadata = json.load(file)
            if not isinstance(metadata, dict):
                raise TypeError("Parsed metadata JSON is not a dictionary.")
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return

    if args.params:
        try:
            params = json.loads(args.params)
            if not isinstance(params, dict):
                raise TypeError("Parsed params JSON is not a dictionary.")
        except json.JSONDecodeError as e:
            print(f"Error parsing params: {str(e)}")
            return
    else:
        params = None
    #clean up params here
    await start_test(data=args.file, test_set=test_set, user_id= args.user_id, params= params, metadata =metadata, retriever_type=args.retriever_type)


if __name__ == "__main__":
    asyncio.run(main())

    # delete_mems = await memory.dynamic_method_call(dynamic_memory_class, 'delete_memories',
    #                                                namespace=test_id)
    # test_load_pipeline = await asyncio.gather(
    #     *(run_load_test_element(test_item,loader_settings, metadata, test_id) for test_item in test_set)
    # )
    #
    # test_eval_pipeline = await asyncio.gather(
    #     *(run_search_eval_element(test_item, test_id) for test_item in test_set)
    # )
    # logging.info("Results of the eval pipeline %s", str(test_eval_pipeline))
    # await add_entity(session, TestOutput(id=test_id, user_id=user_id, test_results=str(test_eval_pipeline)))
    # return test_eval_pipeline

# # Gather and run all tests in parallel
# results = await asyncio.gather(
#     *(run_testo(test, loader_settings, metadata) for test in test_params)
# )
# return results
