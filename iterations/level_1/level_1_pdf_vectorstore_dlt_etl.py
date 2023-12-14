#Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client

import dlt
from langchain import PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import weaviate
import os
import json

import argparse

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.vectorstores import Weaviate
import uuid
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain import OpenAI, LLMMathChain

import os

embeddings = OpenAIEmbeddings()

from deep_translator import (GoogleTranslator)

def _convert_pdf_to_document(path: str = None):

    """Convert a PDF document to a Document object"""
    if path is None:
        raise ValueError("A valid path to the document must be provided.")

    loader = PyPDFLoader(path)
    pages = loader.load_and_split()

    print("PAGES", pages[0])

    # Parse metadata from the folder path
    path_parts = Path(path).parts
    personal_receipts_index = path_parts.index("personal_receipts")
    metadata_parts = path_parts[personal_receipts_index+1:]

    documents = []
    for page in pages:
        translation = GoogleTranslator(source='auto', target='en').translate(text=page.page_content)
        documents.append(
            Document(
                metadata={
                    "title": "Personal Receipt",
                    "country": metadata_parts[1],
                    "year": metadata_parts[0],
                    "author": str(uuid.uuid4()),
                    "source": "/".join(metadata_parts),
                },
                page_content=translation,
            )
        )
    print(documents)

    return documents



def _init_weaviate():
    """Initialize weaviate client and retriever"""
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
    client = weaviate.Client(
        url='https://my-vev-index-o4qitptw.weaviate.network',
        auth_client_secret=auth_config,

        additional_headers={
            "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
        }
    )
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="PDFloader",
        text_key="text",
        attributes=[],
        embedding=embeddings,
        create_schema_if_missing=True,
    )

    return retriever
def load_to_weaviate(document_path=None):
    """Load documents to weaviate"""
    retriever =_init_weaviate()

    docs = _convert_pdf_to_document(document_path)

    return retriever.add_documents(docs)


def get_from_weaviate(query=None, path=None, operator=None, valueText=None):
    """
    Get documents from weaviate.

    Args:
        query (str): The query string.
        path (list): The path for filtering, e.g., ['year'].
        operator (str): The operator for filtering, e.g., 'Equal'.
        valueText (str): The value for filtering, e.g., '2017*'.

    Example:
        get_from_weaviate(query="some query", path=['year'], operator='Equal', valueText='2017*')
    """
    retriever = _init_weaviate()

    # Initial retrieval without filters
    output = retriever.get_relevant_documents(
        query,
        score=True,
    )

    # Apply filters if provided
    if path or operator or valueText:
        # Create the where_filter based on provided parameters
        where_filter = {
            'path': path if path else [],
            'operator': operator if operator else '',
            'valueText': valueText if valueText else ''
        }

        # Retrieve documents with filters applied
        output = retriever.get_relevant_documents(
            query,
            score=True,
            where_filter=where_filter
        )

    return output


def delete_from_weaviate(query=None, filters=None):
    """Delete documents from weaviate, pass dict as filters"""
    """  {
        'path': ['year'],
        'operator': 'Equal',
        'valueText': '2017*'     }"""
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
    client = weaviate.Client(
        url='https://my-vev-index-o4qitptw.weaviate.network',
        auth_client_secret=auth_config,

        additional_headers={
            "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
        }
    )
    client.batch.delete_objects(
        class_name='PDFloader',
        # Same `where` filter as in the GraphQL API
        where={
            'path': ['year'],
            'operator': 'Equal',
            'valueText': '2017*'
        },
    )

    return "Success"


llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4-0613",
        )



def infer_schema_from_text(text: str):
    """Infer schema from text"""

    prompt_ = """ You are a json schema master. Create a JSON schema based on the following data and don't write anything else: {prompt} """

    complete_query = PromptTemplate(
    input_variables=["prompt"],
    template=prompt_,
)

    chain = LLMChain(
        llm=llm, prompt=complete_query, verbose=True
    )
    chain_result = chain.run(prompt=text).strip()

    json_data = json.dumps(chain_result)
    return json_data


def set_data_contract(data, version, date, agreement_id=None, privacy_policy=None, terms_of_service=None, format=None, schema_version=None, checksum=None, owner=None, license=None, validity_start=None, validity_end=None):
    # Creating the generic data contract

    data_contract = {
        "version": version or "",
        "date": date or "",
        "agreement_id": agreement_id or "",
        "privacy_policy": privacy_policy or "",
        "terms_of_service": terms_of_service or "",
        "format": format or "",
        "schema_version": schema_version or "",
        "checksum": checksum or "",
        "owner": owner or "",
        "license": license or "",
        "validity_start": validity_start or "",
        "validity_end": validity_end or "",
        "properties": data  # Adding the given data under the "properties" field
    }



    return data_contract

def create_id_dict(memory_id=None, st_memory_id=None, buffer_id=None):
    """
    Create a dictionary containing IDs for memory, st_memory, and buffer.

    Args:
        memory_id (str): The Memory ID.
        st_memory_id (str): The St_memory ID.
        buffer_id (str): The Buffer ID.

    Returns:
        dict: A dictionary containing the IDs.
    """
    id_dict = {
        "memoryID": memory_id or "",
        "st_MemoryID": st_memory_id or "",
        "bufferID": buffer_id or ""
    }
    return id_dict



def init_buffer(data, version, date, memory_id=None, st_memory_id=None, buffer_id=None, agreement_id=None, privacy_policy=None, terms_of_service=None, format=None, schema_version=None, checksum=None, owner=None, license=None, validity_start=None, validity_end=None, text=None, process=None):
    # Create ID dictionary
    id_dict = create_id_dict(memory_id, st_memory_id, buffer_id)

    # Set data contract
    data_contract = set_data_contract(data, version, date, agreement_id, privacy_policy, terms_of_service, format, schema_version, checksum, owner, license, validity_start, validity_end)

    # Add ID dictionary to properties
    data_contract["properties"]["relations"] = id_dict

    # Infer schema from text and add to properties
    if text:
        schema = infer_schema_from_text(text)
        data_contract["properties"]["schema"] = schema

    if process:
        data_contract["properties"]["process"] = process


    return data_contract


def infer_properties_from_text(text: str):
    """Infer schema properties from text"""

    prompt_ = """ You are a json index master. Create a short JSON index containing the most important data and don't write anything else: {prompt} """

    complete_query = PromptTemplate(
    input_variables=["prompt"],
    template=prompt_,
)

    chain = LLMChain(
        llm=llm, prompt=complete_query, verbose=True
    )
    chain_result = chain.run(prompt=text).strip()
    # json_data = json.dumps(chain_result)
    return chain_result
#
#
# # print(infer_schema_from_text(output[0].page_content))


def load_json_or_infer_schema(file_path, document_path):
    """Load JSON schema from file or infer schema from text"""
    try:
        # Attempt to load the JSON file
        with open(file_path, 'r') as file:
            json_schema = json.load(file)
        return json_schema
    except FileNotFoundError:
        # If the file doesn't exist, run the specified function
        output = _convert_pdf_to_document(path=document_path)
        json_schema = infer_schema_from_text(output[0].page_content)
        return json_schema




def ai_function(prompt=None, json_schema=None):
    """AI function to convert unstructured data to structured data"""
    # Here we define the user prompt and the structure of the output we desire
    # prompt = output[0].page_content

    prompt_msgs = [
        SystemMessage(
            content="You are a world class algorithm converting unstructured data into structured data."
        ),
        HumanMessage(content="Convert unstructured data to structured data:"),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt_ = ChatPromptTemplate(messages=prompt_msgs)
    chain = create_structured_output_chain(json_schema , prompt=prompt_, llm=llm, verbose=True)
    output = chain.run(input = prompt, llm=llm)
    yield output


# Define a base directory if you have one; this could be the directory where your script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def higher_level_thinking():
    """Higher level thinking function to calculate the sum of the price of the tickets from these documents"""

    docs_data = get_from_weaviate(query="Train", path=['year'], operator='Equal', valueText='2017*')
    str_docs_data = str(docs_data)

    llm_math = LLMMathChain.from_llm(llm, verbose=True)
    output = llm_math.run(f"Calculate the sum of the price of the tickets from these documents: {str_docs_data}")

    # data_format = init_buffer(data=output, version="0.0.1", date="2021-09-01")
    yield output
result_higher_level_thinking = higher_level_thinking()
def process_higher_level_thinking(result=None):
    data_format = init_buffer(data=result, version="0.0.1", date="2021-09-01")
    import json
    data_format=json.dumps(data_format)
    yield data_format

document_paths = [
    os.path.join(BASE_DIR, "personal_receipts", "2017", "de", "public_transport", "3ZCCCW.pdf"),
    os.path.join(BASE_DIR, "personal_receipts", "2017", "de", "public_transport", "4GBEC9.pdf")
]



def main(raw_loading, processed_loading,document_paths):
    BASE_DIR = os.getcwd()  # Assuming the current working directory is where the data_processing_script.py is located

    def format_document_paths(base_dir, path):
        # Split the input path and extract the elements
        elements = path.strip("/").split("/")

        # Construct the document_paths list
        document_paths = [os.path.join(base_dir, *elements)]

        return document_paths

    document_paths_ =[format_document_paths(BASE_DIR, path) for path in document_paths][0]
    print(document_paths)

    if raw_loading:
        for document in document_paths_:
            file_path = os.path.join(BASE_DIR, "ticket_schema.json")
            json_schema = load_json_or_infer_schema(file_path, document)
            output = _convert_pdf_to_document(path=document)
            try:
                find_data_in_store = get_from_weaviate(query="Train", path=['year'], operator='Equal', valueText='2017*')

                output = find_data_in_store
                print(output[1])
            except:
                load_to_weaviate(document)

            pipeline = dlt.pipeline(pipeline_name="train_ticket", destination='duckdb', dataset_name='train_ticket_data')
            info = pipeline.run(data=ai_function(output[0].page_content, json_schema))
            print(info)

    elif processed_loading:
        pipeline_processed = dlt.pipeline(pipeline_name="train_ticket_processed", destination='duckdb',
                                          dataset_name='train_ticket_processed_data')
        info = pipeline_processed.run(data=higher_level_thinking())
        print(info)

    else:
        print("Please specify either '--raw_loading' or '--processed_loading' option.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Script")
    parser.add_argument("--raw_loading", action="store_true", help="Load raw document data and perform AI tasks")
    parser.add_argument("--processed_loading", action="store_true",
                        help="Load processed data and run higher-level thinking AI function")
    parser.add_argument("document_paths", nargs="*", help="Paths to the documents to process")

    args = parser.parse_args()

    main(args.raw_loading, args.processed_loading, args.document_paths)


#TO RUN:  python3 level_1_pdf_vectorstore_dlt_etl.py --raw_loading "/personal_receipts/2017/de/public_transport/3ZCCCW.pdf"

