# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client

import dlt
from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import weaviate
import os
import json
import asyncio
from typing import Any, Dict, List, Coroutine
from deep_translator import (GoogleTranslator)
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from pydantic import BaseModel, Field, parse_obj_as
from langchain.memory import VectorStoreRetrieverMemory
from marvin import ai_classifier
from enum import Enum
import marvin
import asyncio
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.schema import Document, SystemMessage, HumanMessage, LLMResult
from langchain.tools import tool
from langchain.vectorstores import Weaviate
import uuid
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from langchain import OpenAI, LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import os

from datetime import datetime
import os
from datetime import datetime
from jinja2 import Template
from langchain import PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.vectorstores import Weaviate
import weaviate
import uuid
import humanize
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


class VectorDB:
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    def __init__(self, user_id: str, index_name: str, memory_id: str, ltm_memory_id: str = '00000',
                 st_memory_id: str = '0000', buffer_id: str = '0000', db_type: str = "pinecone", namespace: str = None):
        self.user_id = user_id
        self.index_name = index_name
        self.db_type = db_type
        self.namespace = namespace
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.st_memory_id = st_memory_id
        self.buffer_id = buffer_id
        # if self.db_type == "pinecone":
        #     self.vectorstore = self.init_pinecone(self.index_name)
        if self.db_type == "weaviate":
            self.init_weaviate(namespace=self.namespace)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        if self.db_type == "weaviate":
            self.init_weaviate_client(namespace=self.namespace)
        else:
            raise ValueError(f"Unsupported VectorDB client type: {db_type}")
        load_dotenv()

    def init_pinecone(self, index_name):
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
        PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        pinecone.Index(index_name)
        vectorstore: Pinecone = Pinecone.from_existing_index(

            index_name=self.index_name,
            embedding=OpenAIEmbeddings(),
            namespace='RESULT'
        )
        return vectorstore

    def init_weaviate_client(self, namespace: str):
        embeddings = OpenAIEmbeddings()
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
        client = weaviate.Client(
            url=os.environ.get('WEAVIATE_URL'),
            auth_client_secret=auth_config,

            additional_headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
            }
        )
        return client

    def init_weaviate(self, namespace: str):
        embeddings = OpenAIEmbeddings()
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
        client = weaviate.Client(
            url=os.environ.get('WEAVIATE_URL'),
            auth_client_secret=auth_config,

            additional_headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
            }
        )
        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=namespace,
            text_key="text",
            attributes=[],
            embedding=embeddings,
            create_schema_if_missing=True,
        )
        return retriever

    async def add_memories(self, observation: str, params: dict = None):
        if self.db_type == "pinecone":
            # Update Pinecone memories here
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index_name, embedding=OpenAIEmbeddings(), namespace=self.namespace
            )
            retriever = vectorstore.as_retriever()
            retriever.add_documents(
                [
                    Document(
                        page_content=observation,
                        metadata={
                            "inserted_at": datetime.now(),
                            "text": observation,
                            "user_id": self.user_id,
                            "version": params.get('version', None) or "",
                            "agreement_id": params.get('agreement_id', None) or "",
                            "privacy_policy": params.get('privacy_policy', None) or "",
                            "terms_of_service": params.get('terms_of_service', None) or "",
                            "format": params.get('format', None) or "",
                            "schema_version": params.get('schema_version', None) or "",
                            "checksum": params.get('checksum', None) or "",
                            "owner": params.get('owner', None) or "",
                            "license": params.get('license', None) or "",
                            "validity_start": params.get('validity_start', None) or "",
                            "validity_end": params.get('validity_end', None) or ""
                        },
                        namespace=self.namespace,
                    )
                ]
            )
        elif self.db_type == "weaviate":
            # Update Weaviate memories here
            print(self.namespace)
            retriever = self.init_weaviate(self.namespace)

            return retriever.add_documents([
                Document(
                    metadata={
                        "text": observation,
                        "user_id": str(self.user_id),
                        "memory_id": str(self.memory_id),
                        "ltm_memory_id": str(self.ltm_memory_id),
                        "st_memory_id": str(self.st_memory_id),
                        "buffer_id": str(self.buffer_id),
                        "version": params.get('version', None) or "",
                        "agreement_id": params.get('agreement_id', None) or "",
                        "privacy_policy": params.get('privacy_policy', None) or "",
                        "terms_of_service": params.get('terms_of_service', None) or "",
                        "format": params.get('format', None) or "",
                        "schema_version": params.get('schema_version', None) or "",
                        "checksum": params.get('checksum', None) or "",
                        "owner": params.get('owner', None) or "",
                        "license": params.get('license', None) or "",
                        "validity_start": params.get('validity_start', None) or "",
                        "validity_end": params.get('validity_end', None) or ""

                        # **source_metadata,
                    },
                    page_content=observation,
                )]
            )

    # def get_pinecone_vectorstore(self, namespace: str) -> pinecone.VectorStore:
    #     return Pinecone.from_existing_index(
    #         index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
    #     )

    async def fetch_memories(self, observation: str, namespace: str, params: dict = None):
        if self.db_type == "pinecone":
            # Fetch Pinecone memories here
            pass
        elif self.db_type == "weaviate":
            # Fetch Weaviate memories here
            """
            Get documents from weaviate.

            Args a json containing:
                query (str): The query string.
                path (list): The path for filtering, e.g., ['year'].
                operator (str): The operator for filtering, e.g., 'Equal'.
                valueText (str): The value for filtering, e.g., '2017*'.

            Example:
                get_from_weaviate(query="some query", path=['year'], operator='Equal', valueText='2017*')
            """
            client = self.init_weaviate_client(self.namespace)

            print(self.namespace)
            print(str(datetime.now()))
            print(observation)
            if namespace is None:
                namespace = self.namespace

            params_user_id = {
                "path": ["user_id"],
                "operator": "Like",
                "valueText": self.user_id
            }

            if params:
                query_output = client.query.get(namespace, ["text"
                    , "user_id"
                    , "memory_id"
                    , "ltm_memory_id"
                    , "st_memory_id"
                    , "buffer_id"
                    , "version",
                                                            "agreement_id",
                                                            "privacy_policy",
                                                            "terms_of_service",
                                                            "format",
                                                            "schema_version",
                                                            "checksum",
                                                            "owner",
                                                            "license",
                                                            "validity_start",
                                                            "validity_end"]).with_where(params).with_additional(
                    ['id', 'creationTimeUnix', 'lastUpdateTimeUnix', "score"]).with_where(params_user_id).do()
                return query_output
            else:
                query_output = client.query.get(namespace, ["text",
                                                            "user_id",
                                                            "memory_id",
                                                            "ltm_memory_id",
                                                            "st_memory_id",
                                                            "buffer_id",
                                                            "version",
                                                            "agreement_id",
                                                            "privacy_policy",
                                                            "terms_of_service",
                                                            "format",
                                                            "schema_version",
                                                            "checksum",
                                                            "owner",
                                                            "license",
                                                            "validity_start",
                                                            "validity_end"
                                                            ]).with_additional(
                    ['id', 'creationTimeUnix', 'lastUpdateTimeUnix', "score"]).with_where(params_user_id).do()
                return query_output

    async def delete_memories(self, params: dict = None):
        client = self.init_weaviate_client(self.namespace)
        if params:
            where_filter = {
                "path": ["id"],
                "operator": "Equal",
                "valueText": params.get('id', None)
            }
            return client.batch.delete_objects(
                class_name=self.namespace,
                # Same `where` filter as in the GraphQL API
                where=where_filter,
            )
        else:
            # Delete all objects

            return client.batch.delete_objects(
                class_name=self.namespace,
                where={
                    'path': ['user_id'],
                    'operator': 'Equal',
                    'valueText': self.user_id
                }
            )

    def update_memories(self, observation, namespace: str, params: dict = None):
        client = self.init_weaviate_client(self.namespace)

        client.data_object.update(
            data_object={
                "text": observation,
                "user_id": str(self.user_id),
                "memory_id": str(self.memory_id),
                "ltm_memory_id": str(self.ltm_memory_id),
                "st_memory_id": str(self.st_memory_id),
                "buffer_id": str(self.buffer_id),
                "version": params.get('version', None) or "",
                "agreement_id": params.get('agreement_id', None) or "",
                "privacy_policy": params.get('privacy_policy', None) or "",
                "terms_of_service": params.get('terms_of_service', None) or "",
                "format": params.get('format', None) or "",
                "schema_version": params.get('schema_version', None) or "",
                "checksum": params.get('checksum', None) or "",
                "owner": params.get('owner', None) or "",
                "license": params.get('license', None) or "",
                "validity_start": params.get('validity_start', None) or "",
                "validity_end": params.get('validity_end', None) or ""

                # **source_metadata,

            },
            class_name="Test",
            uuid=params.get('id', None),
            consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,  # default QUORUM
        )
        return


class SemanticMemory:
    def __init__(self, user_id: str, memory_id: str, ltm_memory_id: str, index_name: str, db_type: str = "weaviate",
                 namespace: str = "SEMANTICMEMORY"):
        # Add any semantic memory-related attributes or setup here
        self.user_id = user_id
        self.index_name = index_name
        self.namespace = namespace
        self.semantic_memory_id = str(uuid.uuid4())
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.vector_db = VectorDB(user_id=user_id, memory_id=self.memory_id, ltm_memory_id=self.ltm_memory_id,
                                  index_name=index_name, db_type=db_type, namespace=self.namespace)
        self.db_type = db_type

    async def _add_memories(self, semantic_memory: str = "None", params: dict = None) -> list[str]:
        """Update semantic memory for the user"""

        if self.db_type == "weaviate":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.create_documents([semantic_memory])
            for text in texts:
                out = await self.vector_db.add_memories(observation=text.page_content, params=params)
                return out

        elif self.db_type == "pinecone":
            pass

    async def _fetch_memories(self, observation: str, params: str = None) -> Coroutine[Any, Any, Any]:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)

        if self.db_type == "weaviate":

            return await self.vector_db.fetch_memories(observation, params)

        elif self.db_type == "pinecone":
            pass

    async def _delete_memories(self, params: str = None) -> Coroutine[Any, Any, Any]:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)

        if self.db_type == "weaviate":

            return await self.vector_db.delete_memories(params=params)

        elif self.db_type == "pinecone":
            pass


class EpisodicMemory:
    def __init__(self, user_id: str, memory_id: str, ltm_memory_id: str, index_name: str, db_type: str = "weaviate",
                 namespace: str = "EPISODICMEMORY"):
        # Add any semantic memory-related attributes or setup here
        self.user_id = user_id
        self.index_name = index_name
        self.namespace = namespace
        self.episodic_memory_id = str(uuid.uuid4())
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.vector_db = VectorDB(user_id=user_id, memory_id=self.memory_id, ltm_memory_id=self.ltm_memory_id,
                                  index_name=index_name, db_type=db_type, namespace=self.namespace)
        self.db_type = db_type

    async def _add_memories(self, observation: str = None, params: dict = None) -> list[str]:
        """Update semantic memory for the user"""

        if self.db_type == "weaviate":
            return await self.vector_db.add_memories(observation=observation, params=params)

        elif self.db_type == "pinecone":
            pass

    def _fetch_memories(self, observation: str, params: str = None) -> Coroutine[Any, Any, Any]:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)

        if self.db_type == "weaviate":

            return self.vector_db.fetch_memories(observation, params)

        elif self.db_type == "pinecone":
            pass

    async def _delete_memories(self, params: str = None) -> Coroutine[Any, Any, Any]:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)

        if self.db_type == "weaviate":

            return await self.vector_db.delete_memories(params=params)

        elif self.db_type == "pinecone":
            pass


class LongTermMemory:
    def __init__(self, user_id: str = "676", memory_id: str = None, index_name: str = None, namespace: str = None,
                 db_type: str = "weaviate"):
        self.user_id = user_id
        self.memory_id = memory_id
        self.ltm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.namespace = namespace
        self.db_type = db_type
        # self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory(user_id=self.user_id, memory_id=self.memory_id,
                                              ltm_memory_id=self.ltm_memory_id, index_name=self.index_name,
                                              db_type=self.db_type)
        self.episodic_memory = EpisodicMemory(user_id=self.user_id, memory_id=self.memory_id,
                                              ltm_memory_id=self.ltm_memory_id, index_name=self.index_name,
                                              db_type=self.db_type)


class ShortTermMemory:
    def __init__(self, user_id: str = "676", memory_id: str = None, index_name: str = None, namespace: str = None,
                 db_type: str = "weaviate"):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id
        self.namespace = namespace
        self.db_type = db_type
        self.stm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.episodic_buffer = EpisodicBuffer(user_id=self.user_id, memory_id=self.memory_id,
                                              index_name=self.index_name, db_type=self.db_type)


class EpisodicBuffer:
    def __init__(self, user_id: str = "676", memory_id: str = None, index_name: str = None,
                 namespace: str = 'EPISODICBUFFER', db_type: str = "weaviate"):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id
        self.namespace = namespace
        self.db_type = db_type
        self.st_memory_id = "blah"
        self.index_name = index_name
        self.llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4-0613",
            callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
        )
        self.llm_base = OpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4-0613"
        )

        # self.vector_db = VectorDB(user_id=user_id, memory_id= self.memory_id, st_memory_id = self.st_memory_id, index_name=index_name, db_type=db_type, namespace=self.namespace)

    # async def infer_schema_from_text(self, text: str):
    #     """Infer schema from text"""
    #
    #     prompt_ = """ You are a json schema master. Create a JSON schema based on the following data and don't write anything else: {prompt} """
    #
    #     complete_query = PromptTemplate(
    #         input_variables=["prompt"],
    #         template=prompt_,
    #     )
    #
    #     chain = LLMChain(
    #         llm=self.llm, prompt=complete_query, verbose=True
    #     )
    #     chain_result = chain.run(prompt=text).strip()
    #
    #     json_data = json.dumps(chain_result)
    #     return json_data

    async def _fetch_memories(self, observation: str, namespace: str) -> str:
        vector_db = VectorDB(user_id=self.user_id, memory_id=self.memory_id, st_memory_id=self.st_memory_id,
                             index_name=self.index_name, db_type=self.db_type, namespace=namespace)

        query = await vector_db.fetch_memories(observation=observation, namespace=namespace)
        return query

    async def _add_memories(self, observation: str, namespace: str, params: dict = None):
        vector_db = VectorDB(user_id=self.user_id, memory_id=self.memory_id, st_memory_id=self.st_memory_id,
                             index_name=self.index_name, db_type=self.db_type, namespace=namespace)

        query = await vector_db.add_memories(observation, params=params)
        return query

    async def _delete_memories(self, params: str = None) -> Coroutine[Any, Any, Any]:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)
        vector_db = VectorDB(user_id=self.user_id, memory_id=self.memory_id, st_memory_id=self.st_memory_id,
                             index_name=self.index_name, db_type=self.db_type, namespace=self.namespace)

        if self.db_type == "weaviate":

            return await vector_db.delete_memories(params=params)

        elif self.db_type == "pinecone":
            pass

    async def freshness(self, observation: str,namespace:str=None) -> list[str]:
        """Freshness - Score between 1 and 5  on how often was the information updated in episodic or semantic memory in the past"""

        memory = Memory(user_id=self.user_id)
        await memory.async_init()

        lookup_value = await memory._fetch_episodic_memory(observation = observation)
        unix_t = lookup_value["data"]["Get"]["EPISODICMEMORY"][0]["_additional"]["lastUpdateTimeUnix"]

        # Convert Unix timestamp to datetime
        last_update_datetime = datetime.fromtimestamp(int(unix_t) / 1000)
        time_difference = datetime.now() - last_update_datetime
        time_difference_text = humanize.naturaltime(time_difference)
        marvin.settings.openai.api_key = os.environ.get('OPENAI_API_KEY')
        @ai_classifier
        class MemoryRoute(Enum):
            """Represents classifer for freshness of memories"""

            data_uploaded_now = "0"
            data_uploaded_very_recently = "1"
            data_uploaded_recently = "2"
            data_uploaded_more_than_a_month_ago = "3"
            data_uploaded_more_than_three_months_ago = "4"
            data_uploaded_more_than_six_months_ago = "5"

        namespace = MemoryRoute(str(time_difference_text))
        return [namespace.value, lookup_value]


    async def frequency(self, observation: str,namespace:str) -> list[str]:
        """Frequency - Score between 1 and 5 on how often was the information processed in episodic memory in the past
           Counts the number of times a memory was accessed in the past and divides it by the total number of memories in the episodic memory """
        client = self.init_weaviate_client(self.namespace)

        memory = Memory(user_id=self.user_id)
        await memory.async_init()

        result_output = await memory._fetch_episodic_memory(observation=observation)
        number_of_relevant_events = len(result_output["data"]["Get"]["EPISODICMEMORY"])
        number_of_total_events = client.query.aggregate( self.namespace).with_meta_count().do()
        frequency = float(number_of_relevant_events) / float(number_of_total_events)
        return [str(frequency), result_output["data"]["Get"]["EPISODICMEMORY"][0]]



    async def relevance(self, observation: str) -> list[str]:
        """Relevance - Score between 1 and 5 on how often was the final information relevant to the user in the past.
           Stored in the episodic memory, mainly to show how well a buffer did the job
           Starts at 1, gets updated based on the user feedback """

        return ["5", "memory"]

    async def saliency(self, observation: str) -> list[str]:
        """Determines saliency by finding relevance between user input and document schema values.
        After finding document schena value relevant for the user, it forms a new query based on the schema value and the user input """

        return ["5", "memory"]

    # @ai_classifier
    # class MemoryRoute(Enum):
    #     """Represents classifer for freshness of memories"""
    #
    #     data_uploaded_now = "0"
    #     data_uploaded_very_recently = "1"
    #     data_uploaded_recently = "2"
    #     data_uploaded_more_than_a_month_ago = "3"
    #     data_uploaded_more_than_three_months_ago = "4"
    #     data_uploaded_more_than_six_months_ago = "5"
    #
    # namespace= MemoryRoute(observation)

    # return ggur

    async def encoding(self, document: str, namespace: str = "EPISODICBUFFER", params:dict=None) -> list[str]:
        """Encoding for the buffer, stores raw data in the buffer
        Note, this is not comp-sci encoding, but rather encoding in the sense of storing the content in the buffer"""
        vector_db = VectorDB(user_id=self.user_id, memory_id=self.memory_id, st_memory_id=self.st_memory_id,
                             index_name=self.index_name, db_type=self.db_type, namespace=namespace)

        query = await vector_db.add_memories(document, params=params)
        return query

    async def available_operations(self) -> list[str]:
        """Determines what operations are available for the user to process PDFs"""

        return ["translate", "structure", "load to database", "load to semantic memory", "load to episodic memory", "load to buffer"]

    async def main_buffer(self, user_input=None, content=None, params=None):

        """AI buffer to understand user PDF query, prioritize memory info and process it based on available operations"""

        # we get a list of available operations for our buffer to consider
        # these operations are what we can do with the data, in the context of PDFs (load, translate, structure, etc)
        list_of_operations = await self.available_operations()

        memory = Memory(user_id=self.user_id)
        await memory.async_init()
        await memory._delete_buffer_memory()


        #we just filter the data here to make sure input is clean
        prompt_filter = ChatPromptTemplate.from_template(
            "Filter and remove uneccessary information that is not relevant in the user query, keep it as original as possbile: {query}")
        chain_filter = prompt_filter | self.llm
        output = await chain_filter.ainvoke({"query": user_input})

        # this part is mostly unfinished but the idea is to apply different algorithms to the data to fetch the most relevant information from the vector stores
        context = []
        if params:

            if "freshness" in params:
                params.get('freshness', None) # get the value of freshness
                freshness = await self.freshness(observation=str(output))
                context.append(freshness)

            elif "frequency" in params:
                params.get('freshness', None)
                frequency = await self.freshness(observation=str(output))
                print("freshness", frequency)
                context.append(frequency)

                #fix this so it actually filters


        else:
            #defaults to semantic search if we don't want to apply algorithms on the vectordb data
            memory = Memory(user_id=self.user_id)
            await memory.async_init()

            lookup_value_episodic = await memory._fetch_episodic_memory(observation=str(output))
            lookup_value_semantic = await memory._fetch_semantic_memory(observation=str(output))
            lookup_value_buffer = await self._fetch_memories(observation=str(output), namespace=self.namespace)


            context.append(lookup_value_buffer)
            context.append(lookup_value_semantic)
            context.append(lookup_value_episodic)

            #copy the context over into the buffer
            #do i need to do it for the episodic + raw data, might make sense
        print( "HERE IS THE CONTEXT", context)
        class BufferRawContextTerms(BaseModel):
            """Schema for documentGroups"""
            semantic_search_term: str = Field(..., description="The search term to use to get relevant input based on user query")
            document_description: str = Field(None, description="The short summary of what the document is about")
            document_relevance: str = Field(None, description="The relevance of the document for the task on the scale from 1 to 5")


        class BufferRawContextList(BaseModel):
            """Buffer raw context processed by the buffer"""
            docs: List[BufferRawContextTerms] = Field(..., description="List of docs")
            user_query: str = Field(..., description="The original user query")


        parser = PydanticOutputParser(pydantic_object=BufferRawContextList)

        prompt = PromptTemplate(
            template="Summarize and create semantic search queries and relevant document summaries for the user query.\n{format_instructions}\nOriginal query is: {query}\n Retrieved context is: {context}",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=user_input,  context=context)

        document_context_result = self.llm_base(_input.to_string())

        document_context_result_parsed = parser.parse(document_context_result)

        print("HERE ARE THE DOCS PARSED AND STRUCTURED",document_context_result_parsed)
        class Task(BaseModel):
            """Schema for an individual task."""
            task_order: str = Field(..., description="The order at which the task needs to be performed")
            task_name: str = Field(None, description="The task that needs to be performed")
            operation: str = Field(None, description="The operation to be performed")
            original_query: str = Field(None, description="Original user query provided")

        class TaskList(BaseModel):
            """Schema for the record containing a list of tasks."""
            tasks: List[Task] = Field(..., description="List of tasks")

        prompt_filter_chunk = f"The raw context data is {str(document_context_result_parsed)} Based on available operations {list_of_operations} determine only the relevant list of steps and operations sequentially based {output}"
        # chain_filter_chunk = prompt_filter_chunk | self.llm.bind(function_call={"TaskList": "tasks"}, functions=TaskList)
        # output_chunk = await chain_filter_chunk.ainvoke({"query": output, "list_of_operations": list_of_operations})
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for decomposing prompts into steps and operations and choosing relevant ones"
            ),
            HumanMessage(content="Decompose based on the following prompt:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
            HumanMessage(content="Tips: Only choose actions that are relevant to the user query and ignore others")

        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(TaskList, self.llm, prompt_, verbose=True)
        from langchain.callbacks import get_openai_callback
        with get_openai_callback() as cb:
            output = await chain.arun(input=prompt_filter_chunk, verbose=True)
            print(cb)
        # output = json.dumps(output)
        my_object = parse_obj_as(TaskList, output)
        print("HERE IS THE OUTPUT", my_object.json())

        data = json.loads(my_object.json())

        # Extract the list of tasks
        tasks_list = data["tasks"]

        result_tasks =[]

        for task in tasks_list:
            class PromptWrapper(BaseModel):
                observation: str = Field(
                    description="observation we want to fetch from vectordb"
                )
            @tool("convert_to_structured", args_schema=PromptWrapper, return_direct=True)
            def convert_to_structured( observation=None, json_schema=None):
                """Convert unstructured data to structured data"""
                BASE_DIR = os.getcwd()
                json_path = os.path.join(BASE_DIR, "schema_registry", "ticket_schema.json")

                def load_json_or_infer_schema(file_path, document_path):
                    """Load JSON schema from file or infer schema from text"""

                    # Attempt to load the JSON file
                    with open(file_path, 'r') as file:
                        json_schema = json.load(file)
                    return json_schema

                json_schema =load_json_or_infer_schema(json_path, None)
                def run_open_ai_mapper(observation=None, json_schema=None):
                    """Convert unstructured data to structured data"""

                    prompt_msgs = [
                        SystemMessage(
                            content="You are a world class algorithm converting unstructured data into structured data."
                        ),
                        HumanMessage(content="Convert unstructured data to structured data:"),
                        HumanMessagePromptTemplate.from_template("{input}"),
                        HumanMessage(content="Tips: Make sure to answer in the correct format"),
                    ]
                    prompt_ = ChatPromptTemplate(messages=prompt_msgs)
                    chain_funct = create_structured_output_chain(json_schema, prompt=prompt_, llm=self.llm, verbose=True)
                    output = chain_funct.run(input=observation, llm=self.llm)
                    return output

                result = run_open_ai_mapper(observation, json_schema)
                return result
            class TranslateText(BaseModel):
                observation: str = Field(
                    description="observation we want to translate"
                )

            @tool("translate_to_de", args_schema=TranslateText, return_direct=True)
            def translate_to_de(observation, args_schema=TranslateText):
                """Translate to English"""
                out = GoogleTranslator(source='auto', target='de').translate(text=observation)
                return out

            agent = initialize_agent(
                llm=self.llm,
                tools=[translate_to_de, convert_to_structured],
                agent=AgentType.OPENAI_FUNCTIONS,

                verbose=True,
            )
            print("HERE IS THE TASK", task)
            output = agent.run(input=task)
            print(output)
            result_tasks.append(task)
            result_tasks.append(output)



        print("HERE IS THE RESULT TASKS", str(result_tasks))


        await self.encoding(str(result_tasks), self.namespace, params=params)



        buffer_result = await self._fetch_memories(observation=str(output), namespace=self.namespace)

        print("HERE IS THE RESULT TASKS", str(buffer_result))


        class EpisodicTask(BaseModel):
            """Schema for an individual task."""
            task_order: str = Field(..., description="The order at which the task needs to be performed")
            task_name: str = Field(None, description="The task that needs to be performed")
            operation: str = Field(None, description="The operation to be performed")
            operation_result: str = Field(None, description="The result of the operation")

        class EpisodicList(BaseModel):
            """Schema for the record containing a list of tasks."""
            tasks: List[EpisodicTask] = Field(..., description="List of tasks")
            start_date: str = Field(..., description="The order at which the task needs to be performed")
            end_date: str = Field(..., description="The order at which the task needs to be performed")
            user_query: str = Field(..., description="The order at which the task needs to be performed")

        parser = PydanticOutputParser(pydantic_object=EpisodicList)

        prompt = PromptTemplate(
            template="Format the result.\n{format_instructions}\nOriginal query is: {query}\n Steps are: {steps}, buffer is: {buffer}",
            input_variables=["query", "steps", "buffer"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=user_input, steps=str(tasks_list), buffer=buffer_result)
        #
        # print("a few things to do like load episodic memory in a structured format")
        #
        # return "a few things to do like load episodic memory in a structured format"

        output = self.llm_base(_input.to_string())

        result_parsing = parser.parse(output)

        print("here is the parsing result", result_parsing)
        memory = Memory(user_id=self.user_id)
        await memory.async_init()
        #
        lookup_value = await memory._add_episodic_memory(observation=str(output), params=params)
        #now we clean up buffer memory

        await memory._delete_buffer_memory()
        return lookup_value


        #load to buffer once is done

        #fetch everything in the current session and load to episodic memory




class Memory:
    load_dotenv()
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    def __init__(self, user_id: str = "676", index_name: str = None, knowledge_source: str = None,
                 knowledge_type: str = None, db_type: str = "weaviate", namespace: str = None) -> None:
        self.user_id = user_id
        self.index_name = index_name
        self.db_type = db_type
        self.knowledge_source = knowledge_source
        self.knowledge_type = knowledge_type
        self.memory_id = str(uuid.uuid4())
        self.long_term_memory = None
        self.short_term_memory = None
        self.namespace = namespace
        load_dotenv()

    # Asynchronous factory function for creating LongTermMemory
    async def async_create_long_term_memory(self, user_id, memory_id, index_name, namespace, db_type):
        # Perform asynchronous initialization steps if needed
        return LongTermMemory(
            user_id=user_id, memory_id=memory_id, index_name=index_name,
            namespace=namespace, db_type=db_type
        )

    async def async_init(self):
        # Asynchronous initialization of LongTermMemory and ShortTermMemory
        self.long_term_memory = await self.async_create_long_term_memory(
            user_id=self.user_id, memory_id=self.memory_id, index_name=self.index_name,
            namespace=self.namespace, db_type=self.db_type
        )

    async def async_create_short_term_memory(self, user_id, memory_id, index_name, db_type):
        # Perform asynchronous initialization steps if needed
        return ShortTermMemory(
            user_id=user_id, memory_id=memory_id, index_name=index_name, db_type=db_type
        )

    async def async_init(self):
        # Asynchronous initialization of LongTermMemory and ShortTermMemory
        self.long_term_memory = await self.async_create_long_term_memory(
            user_id=self.user_id, memory_id=self.memory_id, index_name=self.index_name,
            namespace=self.namespace, db_type=self.db_type
        )
        self.short_term_memory = await self.async_create_short_term_memory(
            user_id=self.user_id, memory_id=self.memory_id, index_name=self.index_name,
            db_type=self.db_type
        )
        # self.short_term_memory = await ShortTermMemory.async_init(
        #     user_id=self.user_id, memory_id=self.memory_id, index_name=self.index_name, db_type=self.db_type
        # )

    async def _add_semantic_memory(self, semantic_memory: str, params: dict = None):
        return await self.long_term_memory.semantic_memory._add_memories(
            semantic_memory=semantic_memory, params=params

        )

    async def _fetch_semantic_memory(self, observation, params):
        return await self.long_term_memory.semantic_memory._fetch_memories(
            observation=observation, params=params

        )

    async def _delete_semantic_memory(self, params: str = None):
        return await self.long_term_memory.semantic_memory._delete_memories(
            params=params
        )

    async def _add_episodic_memory(self, observation: str, params: dict = None):
        return await self.long_term_memory.episodic_memory._add_memories(
            observation=observation, params=params

        )

    async def _fetch_episodic_memory(self, observation, params: str = None):
        return await self.long_term_memory.episodic_memory._fetch_memories(
            observation=observation, params=params
        )

    async def _delete_episodic_memory(self, params: str = None):
        return await self.long_term_memory.episodic_memory._delete_memories(
            params=params
        )

    async def _run_buffer(self, user_input: str, content: str = None, params:str=None):
        return await self.short_term_memory.episodic_buffer.main_buffer(user_input=user_input, content=content, params=params)

    async def _add_buffer_memory(self, user_input: str, namespace: str = None, params: dict = None):
        return await self.short_term_memory.episodic_buffer._add_memories(observation=user_input, namespace=namespace,
                                                                          params=params)

    async def _fetch_buffer_memory(self, user_input: str, namespace: str = None):
        return await self.short_term_memory.episodic_buffer._fetch_memories(observation=user_input, namespace=namespace)

    async def _delete_buffer_memory(self, params: str = None):
        return await self.short_term_memory.episodic_buffer._delete_memories(
            params=params
        )
    async def _available_operations(self):
        return await self.long_term_memory.episodic_buffer._available_operations()

async def main():
    memory = Memory(user_id="123")
    await memory.async_init()
    params = {
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
        "validity_end": "2024-07-31"
    }

    gg = await memory._run_buffer(user_input="i NEED TRANSLATION TO GERMAN ", content="i NEED TRANSLATION TO GERMAN ", params=params)
    print(gg)

    # gg = await memory._delete_buffer_memory()
    # print(gg)

    episodic = """{
        "start_date": "2023-08-23",
        "end_date": "2023-08-30",
        "user_query": "How can I plan a healthy diet?",
        "action_steps": [
            {
                "step_number": 1,
                "description": "Research and gather information about basic principles of a healthy diet."
            },
            {
                "step_number": 2,
                "description": "Create a weekly meal plan that includes a variety of nutritious foods."
            },
            {
                "step_number": 3,
                "description": "Prepare and cook meals according to your meal plan. Include fruits, vegetables, lean proteins, and whole grains."
            }
        ]
    }"""
    #
    # ggur = await memory._add_episodic_memory(observation = episodic, params=params)
    # print(ggur)

    # fff = await memory._fetch_episodic_memory(observation = "healthy diet")
    # print(len(fff["data"]["Get"]["EPISODICMEMORY"]))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

    # bb = agent._update_semantic_memory(semantic_memory="Users core summary")
    # bb = agent._fetch_semantic_memory(observation= "Users core summary", params =    {
    #     "path": ["inserted_at"],
    #     "operator": "Equal",
    #     "valueText": "*2023*"
    # })
    # buffer = agent._run_buffer(user_input="I want to get a schema for my data")
    # print(bb)
    # rrr = {
    #     "path": ["year"],
    #     "operator": "Equal",
    #     "valueText": "2017*"
    # }

