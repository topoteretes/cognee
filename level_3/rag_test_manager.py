import argparse
import json
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from deepeval.metrics.overall_score import OverallScoreMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test, run_test
from gptcache.embedding import openai
from marvin import ai_classifier

from models.sessions import Session
from models.testset import TestSet
from models.testoutput import TestOutput
from models.metadatas import MetaDatas
from models.operation import Operation
from sqlalchemy.orm import sessionmaker
from database.database import engine
from vectorstore_manager import Memory
import uuid
from contextlib import contextmanager

import random
import string
import itertools
import logging
import dotenv
dotenv.load_dotenv()
import openai
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY", "")
@contextmanager
def session_scope(session):
    """Provide a transactional scope around a series of operations."""
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session rollback due to: {str(e)}")
        raise
    finally:
        session.close()

def retrieve_latest_test_case(session, user_id, memory_id):
    """
    Retrieve the most recently created test case from the database filtered by user_id and memory_id.

    Parameters:
    - session (Session): A database session.
    - user_id (int/str): The ID of the user to filter test cases by.
    - memory_id (int/str): The ID of the memory to filter test cases by.

    Returns:
    - Object: The most recent test case attributes filtered by user_id and memory_id, or None if an error occurs.
    """
    try:
        return (
            session.query(TestSet.attributes_list)
            .filter_by(user_id=user_id, memory_id=memory_id)
            .order_by(TestSet.created_at.desc())
            .first()
        )
    except Exception as e:
        logger.error(f"An error occurred while retrieving the latest test case: {str(e)}")
        return None


def add_entity(session, entity):
    """
    Add an entity (like TestOutput, Session, etc.) to the database.

    Parameters:
    - session (Session): A database session.
    - entity (Base): An instance of an SQLAlchemy model.

    Returns:
    - str: A message indicating whether the addition was successful.
    """
    with session_scope(session):
        session.add(entity)
        session.commit()
        return "Successfully added entity"


def retrieve_job_by_id(session, user_id, job_id):
    """
    Retrieve a job by user ID and job ID.

    Parameters:
    - session (Session): A database session.
    - user_id (int/str): The ID of the user.
    - job_id (int/str): The ID of the job to retrieve.

    Returns:
    - Object: The job attributes filtered by user_id and job_id, or None if an error occurs.
    """
    try:
        return (
            session.query(Session.id)
            .filter_by(user_id=user_id, id=job_id)
            .order_by(Session.created_at.desc())
            .first()
        )
    except Exception as e:
        logger.error(f"An error occurred while retrieving the job: {str(e)}")
        return None

def fetch_job_id(session, user_id=None, memory_id=None, job_id=None):
    try:
        return (
            session.query(Session.id)
            .filter_by(user_id=user_id, id=job_id)
            .order_by(Session.created_at.desc())
            .first()
        )
    except Exception as e:
        # Handle exceptions as per your application's requirements.
        print(f"An error occurred: {str(e)}")
        return None


def compare_output(output, expected_output):
    """Compare the output against the expected output."""
    pass




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

    # Default base values
    defaults = {
        'chunk_size': 500,
        'chunk_overlap': 20,
        'similarity_score': 0.5,
        'metadata_variation': 0,
        'search_type': 'hybrid'
    }

    # Update defaults with provided base parameters
    params = {**defaults, **(base_params if base_params is not None else {})}

    default_increments = {
        'chunk_size': 500,
        'chunk_overlap': 10,
        'similarity_score': 0.1,
        'metadata_variation': 1
    }

    # Update default increments with provided increments
    increments = {**default_increments, **(increments if increments is not None else {})}

    # Default ranges
    default_ranges = {
        'chunk_size': 3,
        'chunk_overlap': 3,
        'similarity_score': 3,
        'metadata_variation': 3
    }

    # Update default ranges with provided ranges
    ranges = {**default_ranges, **(ranges if ranges is not None else {})}

    # Generate parameter variant ranges
    param_ranges = {
        key: [params[key] + i * increments.get(key, 1) for i in range(ranges.get(key, 1))]
        for key in ['chunk_size', 'chunk_overlap', 'similarity_score', 'metadata_variation']
    }


    param_ranges['cognitive_architecture'] = ["simple_index", "cognitive_architecture"]
    # Add search_type with possible values
    param_ranges['search_type'] = ['text', 'hybrid', 'bm25', 'generate', 'generate_grouped']

    # Filter param_ranges based on included_params
    if included_params is not None:
        param_ranges = {key: val for key, val in param_ranges.items() if key in included_params}

    # Generate all combinations of parameter variants
    keys = param_ranges.keys()
    values = param_ranges.values()
    param_variants = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return param_variants

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

async def eval_test(query=None, output=None, expected_output=None, context=None):
    # query = "How does photosynthesis work?"
    # output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    # expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    # context = "Biology"
    result_output = await generate_chatgpt_output(query, context)

    test_case = LLMTestCase(
        query=query,
        output=result_output,
        expected_output=expected_output,
        context=context,

    )
    metric = OverallScoreMetric()

    # If you want to run the test
    test_result = run_test(test_case, metrics=[metric], raise_error=False)
    return test_result
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
        """Represents classifier for the data location"""

        DEVICE = "DEVICE"
        URL = "URL"
        DATABASE = "DATABASE"

    return LocationRoute(data_string).name


def dynamic_test_manager(data, test_set=None, user=None, params=None):
    from deepeval.dataset import create_evaluation_query_answer_pairs
    # fetch random chunks from the document
    #feed them to the evaluation pipeline
    dataset = create_evaluation_query_answer_pairs(
        "Python is a great language for mathematical expression and machine learning.")

    return dataset



def generate_letter_uuid(length=8):
    """Generate a random string of uppercase letters with the specified length."""
    letters = string.ascii_uppercase  # A-Z
    return ''.join(random.choice(letters) for _ in range(length))


def fetch_test_set_id(session, user_id, id):
    try:
        return (
            session.query(TestSet.id)
            .filter_by(user_id=user_id, id=id)
            .order_by(TestSet.created_at)
            .desc().first()
        )
    except Exception as e:
        logger.error(f"An error occurred while retrieving the job: {str(e)}")
        return None


async def start_test(data, test_set=None, user_id=None, params=None, job_id=None ,metadata=None):

    Session = sessionmaker(bind=engine)
    session = Session()


    memory = Memory.create_memory(user_id, session, namespace="SEMANTICMEMORY")

    job_id = fetch_job_id(session, user_id = user_id,job_id =job_id)
    test_set_id = fetch_test_set_id(session, user_id=user_id, id=job_id)
    if job_id is None:
        job_id = str(uuid.uuid4())
        logging.info("we are adding a new job ID")
        add_entity(session, Operation(id = job_id, user_id = user_id))
    if test_set_id is None:
        test_set_id = str(uuid.uuid4())
        add_entity(session, TestSet(id = test_set_id, user_id = user_id, content = str(test_set)))



    if params is None:

        data_format = data_format_route(data)
        data_location = data_location_route(data)
        test_params = generate_param_variants(  included_params=['chunk_size', 'chunk_overlap', 'similarity_score'])



    loader_settings =  {
    "format": f"{data_format}",
    "source": f"{data_location}",
    "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    }

    for test in test_params:
        test_id = str(generate_letter_uuid()) + "_" + "SEMANTICEMEMORY"

        # Adding a memory instance
        memory.add_memory_instance("ExampleMemory")

        # Managing memory attributes
        existing_user = Memory.check_existing_user(user_id, session)
        print("here is the existing user", existing_user)
        memory.manage_memory_attributes(existing_user)

        test_class = test_id + "_class"
        # memory.test_class

        memory.add_dynamic_memory_class(test_id.lower(), test_id)
        dynamic_memory_class = getattr(memory, test_class.lower(), None)



        if dynamic_memory_class is not None:
            memory.add_method_to_class(dynamic_memory_class, 'add_memories')
        else:
            print(f"No attribute named {test_class.lower()} in memory.")

        if dynamic_memory_class is not None:
            memory.add_method_to_class(dynamic_memory_class, 'fetch_memories')
        else:
            print(f"No attribute named {test_class.lower()} in memory.")

        print(f"Trying to access: {test_class.lower()}")
        print("Available memory classes:", memory.list_memory_classes())

        print(f"Trying to check: ", test)
        loader_settings.update(test)
        load_action = await memory.dynamic_method_call(dynamic_memory_class, 'add_memories',
                                               observation='some_observation', params=metadata, loader_settings=loader_settings)
        loader_settings = {key: value for key, value in loader_settings.items() if key not in test}



        test_result_collection =[]

        for test in test_set:
            retrieve_action = await memory.dynamic_method_call(dynamic_memory_class, 'fetch_memories',
                                                               observation=test["question"])

            test_results = await eval_test( query=test["question"], expected_output=test["answer"], context= str(retrieve_action))
            test_result_collection.append(test_results)

            print(test_results)
        if dynamic_memory_class is not None:
            memory.add_method_to_class(dynamic_memory_class, 'delete_memories')
        else:
            print(f"No attribute named {test_class.lower()} in memory.")
        delete_mems = await memory.dynamic_method_call(dynamic_memory_class, 'delete_memories',
                                                       namespace =test_id)

        print(test_result_collection)

        add_entity(session, TestOutput(id=test_id, user_id=user_id, test_results=str(test_result_collection)))

async def main():
    #
    # params = {
    #     "version": "1.0",
    #     "agreement_id": "AG123456",
    #     "privacy_policy": "https://example.com/privacy",
    #     "terms_of_service": "https://example.com/terms",
    #     "format": "json",
    #     "schema_version": "1.1",
    #     "checksum": "a1b2c3d4e5f6",
    #     "owner": "John Doe",
    #     "license": "MIT",
    #     "validity_start": "2023-08-01",
    #     "validity_end": "2024-07-31",
    # }
    #
    # test_set = [
    #     {
    #         "question": "Who is the main character in 'The Call of the Wild'?",
    #         "answer": "Buck"
    #     },
    #     {
    #         "question": "Who wrote 'The Call of the Wild'?",
    #         "answer": "Jack London"
    #     },
    #     {
    #         "question": "Where does Buck live at the start of the book?",
    #         "answer": "In the Santa Clara Valley, at Judge Miller’s place."
    #     },
    #     {
    #         "question": "Why is Buck kidnapped?",
    #         "answer": "He is kidnapped to be sold as a sled dog in the Yukon during the Klondike Gold Rush."
    #     },
    #     {
    #         "question": "How does Buck become the leader of the sled dog team?",
    #         "answer": "Buck becomes the leader after defeating the original leader, Spitz, in a fight."
    #     }
    # ]
    # result = await start_test("https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf", test_set=test_set, user_id="666", params=None, metadata=params)
    #
    parser = argparse.ArgumentParser(description="Run tests against a document.")
    parser.add_argument("--url", required=True, help="URL of the document to test.")
    parser.add_argument("--test_set", required=True, help="Path to JSON file containing the test set.")
    parser.add_argument("--user_id", required=True, help="User ID.")
    parser.add_argument("--params", help="Additional parameters in JSON format.")
    parser.add_argument("--metadata", required=True, help="Path to JSON file containing metadata.")

    args = parser.parse_args()

    with open(args.test_set, "r") as file:
        test_set = json.load(file)

    with open(args.metadata, "r") as file:
        metadata = json.load(file)

    if args.params:
        params = json.loads(args.params)
    else:
        params = None

    await start_test(args.url, test_set, args.user_id, params, metadata)
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())




