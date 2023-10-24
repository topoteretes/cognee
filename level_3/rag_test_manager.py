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
from database.database import engine  # Ensure you have database engine defined somewhere
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
from vectordb.basevectordb import  BaseMemory
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
            .order_by(TestSet.created_at).first()
        )
        return result.scalar_one_or_none()  # scalar_one_or_none() is a non-blocking call
    except Exception as e:
        logging.error(f"An error occurred while retrieving the latest test case: {str(e)}")
        return None

async def add_entity(session, entity):
    async with session_scope(session) as s:  # Use your async session_scope
        s.add(entity)  # No need to commit; session_scope takes care of it
        s.commit()

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
            .order_by(Session.created_at).first()
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None


async def fetch_test_set_id(session, user_id, id):
    try:
        # Await the execution of the query and fetching of the result
        result = await session.execute(
            session.query(TestSet.id)
            .filter_by(user_id=user_id, id=id)
            .order_by(TestSet.created_at).desc().first()
        )
        return result.scalar_one_or_none()  # scalar_one_or_none() is a non-blocking call
    except Exception as e:
        logging.error(f"An error occurred while retrieving the test set: {str(e)}")
        return None

# Adding "embeddings" to the parameter variants function

def generate_param_variants(base_params=None, increments=None, ranges=None, included_params=None):
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
        'chunk_size': 250,
        'chunk_overlap': 20,
        'similarity_score': 0.5,
        'metadata_variation': 0,
        'search_type': 'hybrid',
        'embeddings': 'openai'  # Default value added for 'embeddings'
    }

    # Update defaults with provided base parameters
    params = {**defaults, **(base_params or {})}

    default_increments = {
        'chunk_size': 150,
        'chunk_overlap': 10,
        'similarity_score': 0.1,
        'metadata_variation': 1
    }

    # Update default increments with provided increments
    increments = {**default_increments, **(increments or {})}

    # Default ranges
    default_ranges = {
        'chunk_size': 2,
        'chunk_overlap': 2,
        'similarity_score': 2,
        'metadata_variation': 2
    }

    # Update default ranges with provided ranges
    ranges = {**default_ranges, **(ranges or {})}

    # Generate parameter variant ranges
    param_ranges = {
        key: [params[key] + i * increments.get(key, 1) for i in range(ranges.get(key, 1))]
        for key in ['chunk_size', 'chunk_overlap', 'similarity_score', 'metadata_variation']
    }

    # Add search_type and embeddings with possible values
    param_ranges['search_type'] = ['text', 'hybrid', 'bm25', 'generate', 'generate_grouped']
    param_ranges['embeddings'] = ['openai', 'cohere', 'huggingface']  # Added 'embeddings' values

    # Filter param_ranges based on included_params
    if included_params is not None:
        param_ranges = {key: val for key, val in param_ranges.items() if key in included_params}

    # Generate all combinations of parameter variants
    keys = param_ranges.keys()
    values = param_ranges.values()
    param_variants = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return param_variants


# Generate parameter variants and display a sample of the generated combinations


async def generate_chatgpt_output(query:str, context:str=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": f"{context}"},
            {"role": "user", "content": query}
        ]
    )
    llm_output = response.choices[0].message.content
    # print(llm_output)
    return llm_output

async def eval_test(query=None, output=None, expected_output=None, context=None, synthetic_test_set=False):

    result_output = await generate_chatgpt_output(query, context)

    if synthetic_test_set:
        test_case = synthetic_test_set
    else:

        test_case = LLMTestCase(
            query=query,
            output=result_output,
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
            "context": test_result.context
        }

    test_result_dict =[]
    for test in test_result:
        test_result_it = test_result_to_dict(test)
        test_result_dict.append(test_result_it)
    return test_result_dict
    # You can also inspect the test result class
    # print(test_result)



def data_format_route( data_string: str):
    @ai_classifier
    class FormatRoute(Enum):
        """Represents classifier for the data format"""


        PDF = "PDF"
        UNSTRUCTURED_WEB = "UNSTRUCTURED_WEB"
        GITHUB = "GITHUB"
        TEXT = "TEXT"
        CSV = "CSV"
        WIKIPEDIA = "WIKIPEDIA"

    return FormatRoute(data_string).name


def data_location_route(data_string: str):
    @ai_classifier
    class LocationRoute(Enum):
        """Represents classifier for the data location, if it is device, or database connections string or URL """

        DEVICE = "file_path_starting_with_.data_or_containing_it"
        # URL = "url starting with http or https"
        DATABASE = "database_name_like_postgres_or_mysql"

    return LocationRoute(data_string).name


def dynamic_test_manager(context=None):
    from deepeval.dataset import create_evaluation_query_answer_pairs
    # fetch random chunks from the document
    #feed them to the evaluation pipeline
    dataset = create_evaluation_query_answer_pairs(openai_api_key=os.environ.get("OPENAI_API_KEY"), context= context ,n=10)

    return dataset



def generate_letter_uuid(length=8):
    """Generate a random string of uppercase letters with the specified length."""
    letters = string.ascii_uppercase  # A-Z
    return ''.join(random.choice(letters) for _ in range(length))

async def start_test(data, test_set=None, user_id=None, params=None, job_id=None, metadata=None, generate_test_set=False, only_llm_context=False):


    async with session_scope(session=AsyncSessionLocal()) as session:


        job_id = await fetch_job_id(session, user_id=user_id, job_id=job_id)
        test_set_id = await fetch_test_set_id(session, user_id=user_id, id=job_id)
        memory = await Memory.create_memory(user_id, session, namespace="SEMANTICMEMORY")
        await memory.add_memory_instance("ExampleMemory")
        existing_user = await Memory.check_existing_user(user_id, session)

        if job_id is None:
            job_id = str(uuid.uuid4())
            await add_entity(session, Operation(id=job_id, user_id=user_id))

        if test_set_id is None:
            test_set_id = str(uuid.uuid4())
            await add_entity(session, TestSet(id=test_set_id, user_id=user_id, content=str(test_set)))

        if params is None:
            data_format = data_format_route(data)  # Assume data_format_route is predefined
            logging.info("Data format is %s", data_format)
            data_location = data_location_route(data)
            logging.info("Data location is %s",data_location)# Assume data_location_route is predefined
            test_params = generate_param_variants(
                included_params=['chunk_size'])

        print("Here are the test params", str(test_params))

        loader_settings = {
            "format": f"{data_format}",
            "source": f"{data_location}",
            "path": data
        }


        async def run_test(test, loader_settings, metadata, test_id=None,only_llm_context=False):

            if test_id is None:
                test_id = str(generate_letter_uuid()) + "_" +"SEMANTICMEMORY"
            await memory.manage_memory_attributes(existing_user)
            test_class = test_id + "_class"
            await memory.add_dynamic_memory_class(test_id.lower(), test_id)
            dynamic_memory_class = getattr(memory, test_class.lower(), None)
            methods_to_add = ['add_memories', 'fetch_memories', 'delete_memories']

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
            test_class = test_id + "_class"
            dynamic_memory_class = getattr(memory, test_class.lower(), None)
            async def run_load_test_element( loader_settings=loader_settings, metadata=metadata, test_id=test_id, test_set=test_set):

                print(f"Trying to access: {test_class.lower()}")
                await memory.dynamic_method_call(dynamic_memory_class, 'add_memories',
                                                               observation='Observation loaded', params=metadata,
                                                               loader_settings=loader_settings)
                return "Loaded test element"
            async def run_search_element(test_item, test_id):
                retrieve_action = await memory.dynamic_method_call(dynamic_memory_class, 'fetch_memories',
                                                                   observation=str(test_item["question"]))
                print("Here is the test result", str(retrieve_action["data"]['Get'][test_id][0]["text"]))
                return retrieve_action["data"]['Get'][test_id][0]["text"]

            async def run_eval(test_item, search_result):
                test_eval= await eval_test(query=test_item["question"], expected_output=test_item["answer"],
                                              context=str(search_result))
                return test_eval
            async def run_generate_test_set( test_id):

                test_class = test_id + "_class"
                # await memory.add_dynamic_memory_class(test_id.lower(), test_id)
                dynamic_memory_class = getattr(memory, test_class.lower(), None)
                print(dynamic_memory_class)
                retrieve_action = await memory.dynamic_method_call(dynamic_memory_class, 'fetch_memories',
                                                                   observation="Generate a short summary of this document",
                                                                   search_type="generative")
                return dynamic_test_manager(retrieve_action)
            test_eval_pipeline =[]


            if only_llm_context:
                for test_qa in test_set:
                    context=""
                    test_result = await run_eval(test_qa, context)
                    test_eval_pipeline.append(test_result)
            if generate_test_set is True:
                synthetic_test_set = run_generate_test_set(test_id)
            else:
                pass
            if test_set:
                logging.info("Loading and evaluating test set")
                await run_load_test_element(loader_settings, metadata, test_id, test_set)
                for test_qa in test_set:
                    result = await run_search_element(test_qa, test_id)
                    test_result = await run_eval(test_qa, result)
                    test_eval_pipeline.append( test_result)

            else:
                pass

            await memory.dynamic_method_call(dynamic_memory_class, 'delete_memories',
                                                           namespace=test_id)

            return test_id, test_eval_pipeline

        results = []

        if only_llm_context:
            test_id, result = await run_test(test=None, loader_settings=loader_settings, metadata=metadata,
                                             only_llm_context=only_llm_context)
            results.append(result)

        for param in test_params:
            test_id, result = await run_test(param, loader_settings, metadata, only_llm_context=only_llm_context)
            results.append(result)


        for result_list in results[0]:
            for result in result_list:
                print("Here is one result", result)
                await add_entity(session, TestOutput(id=test_id, user_id=user_id, test_results=result['success'], test_score=str(result['score']), test_metric_name=result['metric_name'], test_query=result['query'], test_output=result['output'], test_expected_output=str(['expected_output']), test_context=result['context'][0]))

        print(results)

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
            "answer": "Buck"
        },
        {
            "question": "Who wrote 'The Call of the Wild'?",
            "answer": "Jack London"
        },
        {
            "question": "Where does Buck live at the start of the book?",
            "answer": "In the Santa Clara Valley, at Judge Miller’s place."
        },
        {
            "question": "Why is Buck kidnapped?",
            "answer": "He is kidnapped to be sold as a sled dog in the Yukon during the Klondike Gold Rush."
        },
        {
            "question": "How does Buck become the leader of the sled dog team?",
            "answer": "Buck becomes the leader after defeating the original leader, Spitz, in a fight."
        }
    ]
    # "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    #http://public-library.uk/ebooks/59/83.pdf
    result = await start_test(".data/3ZCCCW.pdf", test_set=test_set, user_id="677", params=None, metadata=metadata)
    #
    # parser = argparse.ArgumentParser(description="Run tests against a document.")
    # parser.add_argument("--url", required=True, help="URL of the document to test.")
    # parser.add_argument("--test_set", required=True, help="Path to JSON file containing the test set.")
    # parser.add_argument("--user_id", required=True, help="User ID.")
    # parser.add_argument("--params", help="Additional parameters in JSON format.")
    # parser.add_argument("--metadata", required=True, help="Path to JSON file containing metadata.")
    # parser.add_argument("--generate_test_set", required=True, help="Make a test set.")
    # parser.add_argument("--only_llm_context", required=True, help="Do a test only within the existing LLM context")
    # args = parser.parse_args()
    #
    # try:
    #     with open(args.test_set, "r") as file:
    #         test_set = json.load(file)
    #         if not isinstance(test_set, list):  # Expecting a list
    #             raise TypeError("Parsed test_set JSON is not a list.")
    # except Exception as e:
    #     print(f"Error loading test_set: {str(e)}")
    #     return
    #
    # try:
    #     with open(args.metadata, "r") as file:
    #         metadata = json.load(file)
    #         if not isinstance(metadata, dict):
    #             raise TypeError("Parsed metadata JSON is not a dictionary.")
    # except Exception as e:
    #     print(f"Error loading metadata: {str(e)}")
    #     return
    #
    # if args.params:
    #     try:
    #         params = json.loads(args.params)
    #         if not isinstance(params, dict):
    #             raise TypeError("Parsed params JSON is not a dictionary.")
    #     except json.JSONDecodeError as e:
    #         print(f"Error parsing params: {str(e)}")
    #         return
    # else:
    #     params = None
    # #clean up params here
    # await start_test(args.url, test_set, args.user_id, params=None, metadata=metadata)
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