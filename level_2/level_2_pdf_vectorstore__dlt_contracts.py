# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import json
from enum import Enum
from io import BytesIO
from typing import Dict, List, Union

import marvin
import requests
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.tools import tool
from marvin import ai_classifier
from pydantic import parse_obj_as

load_dotenv()
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Optional

import tracemalloc

tracemalloc.start()

import os
from datetime import datetime
from langchain import PromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import Document, SystemMessage, HumanMessage
import uuid
import humanize
import weaviate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
marvin.settings.openai.api_key = os.environ.get("OPENAI_API_KEY")

# class MyCustomSyncHandler(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")
#
#
# class MyCustomAsyncHandler(AsyncCallbackHandler):
#     """Async callback handler that can be used to handle callbacks from langchain."""
#
#     async def on_llm_start(
#             self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
#     ) -> None:
#         """Run when chain starts running."""
#         print("zzzz....")
#         await asyncio.sleep(0.3)
#         class_name = serialized["name"]
#         print("Hi! I just woke up. Your llm is starting")
#
#     async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
#         """Run when chain ends running."""
#         print("zzzz....")
#         await asyncio.sleep(0.3)
#         print("Hi! I just woke up. Your llm is ending")
#
#


# Assuming OpenAIEmbeddings and other necessary imports are available

# Default Values
LTM_MEMORY_ID_DEFAULT = "00000"
ST_MEMORY_ID_DEFAULT = "0000"
BUFFER_ID_DEFAULT = "0000"


class VectorDBFactory:
    def create_vector_db(
        self,
        user_id: str,
        index_name: str,
        memory_id: str,
        ltm_memory_id: str = LTM_MEMORY_ID_DEFAULT,
        st_memory_id: str = ST_MEMORY_ID_DEFAULT,
        buffer_id: str = BUFFER_ID_DEFAULT,
        db_type: str = "pinecone",
        namespace: str = None,
    ):
        db_map = {"pinecone": PineconeVectorDB, "weaviate": WeaviateVectorDB}

        if db_type in db_map:
            return db_map[db_type](
                user_id,
                index_name,
                memory_id,
                ltm_memory_id,
                st_memory_id,
                buffer_id,
                namespace,
            )

        raise ValueError(f"Unsupported database type: {db_type}")


class VectorDB:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    def __init__(
        self,
        user_id: str,
        index_name: str,
        memory_id: str,
        ltm_memory_id: str = LTM_MEMORY_ID_DEFAULT,
        st_memory_id: str = ST_MEMORY_ID_DEFAULT,
        buffer_id: str = BUFFER_ID_DEFAULT,
        namespace: str = None,
    ):
        self.user_id = user_id
        self.index_name = index_name
        self.namespace = namespace
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.st_memory_id = st_memory_id
        self.buffer_id = buffer_id


class PineconeVectorDB(VectorDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_pinecone(self.index_name)

    def init_pinecone(self, index_name):
        # Pinecone initialization logic
        pass


class WeaviateVectorDB(VectorDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_weaviate(self.namespace)

    def init_weaviate(self, namespace: str):
        # Weaviate initialization logic
        embeddings = OpenAIEmbeddings()
        auth_config = weaviate.auth.AuthApiKey(
            api_key=os.environ.get("WEAVIATE_API_KEY")
        )
        client = weaviate.Client(
            url=os.environ.get("WEAVIATE_URL"),
            auth_client_secret=auth_config,
            additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")},
        )
        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=namespace,
            text_key="text",
            attributes=[],
            embedding=embeddings,
            create_schema_if_missing=True,
        )
        return retriever  # If this is part of the initialization, call it here.

    def init_weaviate_client(self, namespace: str):
        # Weaviate client initialization logic
        auth_config = weaviate.auth.AuthApiKey(
            api_key=os.environ.get("WEAVIATE_API_KEY")
        )
        client = weaviate.Client(
            url=os.environ.get("WEAVIATE_URL"),
            auth_client_secret=auth_config,
            additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")},
        )
        return client

    def _document_loader(self, observation: str, loader_settings: dict):
        # Create an in-memory file-like object for the PDF content

        if loader_settings.get("format") == "PDF":

            if loader_settings.get("source") == "url":
                pdf_response = requests.get(loader_settings["path"])
                pdf_stream = BytesIO(pdf_response.content)
                contents = pdf_stream.read()
                tmp_location = os.path.join("/tmp", "tmp.pdf")
                with open(tmp_location, "wb") as tmp_file:
                    tmp_file.write(contents)

                # Process the PDF using PyPDFLoader
                loader = PyPDFLoader(tmp_location)
                # adapt this for different chunking strategies
                pages = loader.load_and_split()
                return pages

            if loader_settings.get("source") == "file":
                # Process the PDF using PyPDFLoader
                # might need adapting for different loaders + OCR
                # need to test the path
                loader = PyPDFLoader(loader_settings["path"])
                pages = loader.load_and_split()

                return pages
        else:
            # Process the text by just loading the base text
            return observation


    async def add_memories(
        self, observation: str, loader_settings: dict = None, params: dict = None ,namespace:str=None
    ):
        # Update Weaviate memories here
        print(self.namespace)
        if namespace is None:
            namespace = self.namespace
        retriever = self.init_weaviate(namespace)

        def _stuct(observation, params):
            """Utility function to not repeat metadata structure"""
            # needs smarter solution, like dynamic generation of metadata
            return [
            Document(
                metadata={
                    # "text": observation,
                    "user_id": str(self.user_id),
                    "memory_id": str(self.memory_id),
                    "ltm_memory_id": str(self.ltm_memory_id),
                    "st_memory_id": str(self.st_memory_id),
                    "buffer_id": str(self.buffer_id),
                    "version": params.get("version", None) or "",
                    "agreement_id": params.get("agreement_id", None) or "",
                    "privacy_policy": params.get("privacy_policy", None) or "",
                    "terms_of_service": params.get("terms_of_service", None) or "",
                    "format": params.get("format", None) or "",
                    "schema_version": params.get("schema_version", None) or "",
                    "checksum": params.get("checksum", None) or "",
                    "owner": params.get("owner", None) or "",
                    "license": params.get("license", None) or "",
                    "validity_start": params.get("validity_start", None) or "",
                    "validity_end": params.get("validity_end", None) or ""
                    # **source_metadata,
                },
                page_content=observation,
            )
        ]

        if loader_settings:
            # Load the document
            document = self._document_loader(observation, loader_settings)
            print("DOC LENGTH", len(document))
            for doc in document:
                document_to_load = _stuct(doc.page_content, params)
                retriever.add_documents(
                    document_to_load
                )

        return retriever.add_documents(
            _stuct(observation, params)
        )

    async def fetch_memories(
        self, observation: str, namespace: str, params: dict = None
    ):
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
            "valueText": self.user_id,
        }

        if params:
            query_output = (
                client.query.get(
                    namespace,
                    [
                        # "text",
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
                        "validity_end",
                    ],
                )
                .with_where(params)
                .with_near_text({"concepts": [observation]})
                .with_additional(
                    ["id", "creationTimeUnix", "lastUpdateTimeUnix", "score",'distance']
                )
                .with_where(params_user_id)
                .do()
            )
            return query_output
        else:
            query_output = (
                client.query.get(
                    namespace,

                    [
                        "text",
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
                        "validity_end",
                    ],
                )
                .with_additional(
                    ["id", "creationTimeUnix", "lastUpdateTimeUnix", "score", 'distance']
                )
                .with_hybrid(
                    query=observation,
                )
                .with_autocut(1)
                .with_where(params_user_id)
                .do()
            )
            return query_output

    async def delete_memories(self, params: dict = None):
        client = self.init_weaviate_client(self.namespace)
        if params:
            where_filter = {
                "path": ["id"],
                "operator": "Equal",
                "valueText": params.get("id", None),
            }
            return client.batch.delete_objects(
                class_name=self.namespace,
                # Same `where` filter as in the GraphQL API
                where=where_filter,
            )
        else:
            # Delete all objects
            print("HERE IS THE USER ID", self.user_id)
            return client.batch.delete_objects(
                class_name=self.namespace,
                where={
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueText": self.user_id,
                },
            )

    def update_memories(self, observation, namespace: str, params: dict = None):
        client = self.init_weaviate_client(self.namespace)

        client.data_object.update(
            data_object={
                # "text": observation,
                "user_id": str(self.user_id),
                "memory_id": str(self.memory_id),
                "ltm_memory_id": str(self.ltm_memory_id),
                "st_memory_id": str(self.st_memory_id),
                "buffer_id": str(self.buffer_id),
                "version": params.get("version", None) or "",
                "agreement_id": params.get("agreement_id", None) or "",
                "privacy_policy": params.get("privacy_policy", None) or "",
                "terms_of_service": params.get("terms_of_service", None) or "",
                "format": params.get("format", None) or "",
                "schema_version": params.get("schema_version", None) or "",
                "checksum": params.get("checksum", None) or "",
                "owner": params.get("owner", None) or "",
                "license": params.get("license", None) or "",
                "validity_start": params.get("validity_start", None) or "",
                "validity_end": params.get("validity_end", None) or ""
                # **source_metadata,
            },
            class_name="Test",
            uuid=params.get("id", None),
            consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,  # default QUORUM
        )
        return


class BaseMemory:
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str,
        namespace: str,
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.index_name = index_name
        self.namespace = namespace
        self.memory_type_id = str(uuid.uuid4())
        self.db_type = db_type
        factory = VectorDBFactory()
        self.vector_db = factory.create_vector_db(
            self.user_id,
            self.index_name,
            self.memory_id,
            db_type=self.db_type,
            namespace=self.namespace,
        )

    def init_client(self, namespace: str):
        if self.db_type == "weaviate":
            return self.vector_db.init_weaviate_client(namespace)

    async def add_memories(
        self,
        observation: Optional[str] = None,
        loader_settings: dict = None,
        params: Optional[dict] = None,
        namespace: Optional[str] = None,
    ):
        if self.db_type == "weaviate":
            return await self.vector_db.add_memories(
                observation=observation, loader_settings=loader_settings, params=params, namespace=namespace
            )
        # Add other db_type conditions if necessary

    async def fetch_memories(
        self,
        observation: str,
        params: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        if self.db_type == "weaviate":
            return await self.vector_db.fetch_memories(
                observation=observation, params=params, namespace=namespace
            )

    async def delete_memories(self, params: Optional[str] = None):
        if self.db_type == "weaviate":
            return await self.vector_db.delete_memories(params)

    # Additional methods for specific Memory can be added here


class SemanticMemory(BaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__(
            user_id, memory_id, index_name, db_type, namespace="SEMANTICMEMORY"
        )


class EpisodicMemory(BaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__(
            user_id, memory_id, index_name, db_type, namespace="EPISODICMEMORY"
        )


class EpisodicBuffer(BaseMemory):
    def __init__(
        self,
        user_id: str,
        memory_id: Optional[str],
        index_name: Optional[str],
        db_type: str = "weaviate",
    ):
        super().__init__(
            user_id, memory_id, index_name, db_type, namespace="BUFFERMEMORY"
        )

        self.st_memory_id = "blah"
        self.llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4-0613",
            # callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
        )
        self.llm_base = OpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4-0613",
        )

    async def memory_route(self, text_time_diff: str):
        @ai_classifier
        class MemoryRoute(Enum):
            """Represents classifer for freshness of memories"""

            data_uploaded_now = "1"
            data_uploaded_very_recently = "0.9"
            data_uploaded_recently = "0.7"
            data_uploaded_more_than_a_month_ago = "0.5"
            data_uploaded_more_than_three_months_ago = "0.3"
            data_uploaded_more_than_six_months_ago = "0.1"

        namespace = MemoryRoute(str(text_time_diff))

        return namespace

    async def freshness(self, observation: str, namespace: str = None) -> list[str]:
        """Freshness - Score between 0 and 1  on how often was the information updated in episodic or semantic memory in the past"""

        lookup_value = await self.fetch_memories(
            observation=observation, namespace=namespace
        )
        unix_t = lookup_value["data"]["Get"]["EPISODICMEMORY"][0]["_additional"][
            "lastUpdateTimeUnix"
        ]

        # Convert Unix timestamp to datetime
        last_update_datetime = datetime.fromtimestamp(int(unix_t) / 1000)
        time_difference = datetime.now() - last_update_datetime
        time_difference_text = humanize.naturaltime(time_difference)
        namespace = await self.memory_route(str(time_difference_text))
        return [namespace.value, lookup_value]

    async def frequency(self, observation: str, namespace: str) -> list[str]:
        """Frequency - Score between 0 and 1 on how often was the information processed in episodic memory in the past
        Counts the number of times a memory was accessed in the past and divides it by the total number of memories in the episodic memory
        """
        weaviate_client = self.init_client(namespace=namespace)

        result_output = await self.fetch_memories(
            observation=observation, params=None, namespace=namespace
        )
        number_of_relevant_events = len(result_output["data"]["Get"]["EPISODICMEMORY"])
        number_of_total_events = (
            weaviate_client.query.aggregate(namespace).with_meta_count().do()
        )
        frequency = float(number_of_relevant_events) / float(
            number_of_total_events["data"]["Aggregate"]["EPISODICMEMORY"][0]["meta"][
                "count"
            ]
        )
        return [str(frequency), result_output["data"]["Get"]["EPISODICMEMORY"][0]]

    async def relevance(self, observation: str, namespace: str) -> list[str]:
        """Relevance - Score between 0 and 1 on how often was the final information relevant to the user in the past.
        Stored in the episodic memory, mainly to show how well a buffer did the job
        Starts at 0, gets updated based on the user feedback"""

        return ["0", "memory"]

    async def saliency(self, observation: str, namespace=None) -> list[str]:
        """Determines saliency by scoring the set of retrieved documents against each other and trying to determine saliency
        """
        class SaliencyRawList(BaseModel):
            """Schema for documentGroups"""
            original_document: str = Field(
                ...,
                description="The original document retrieved from the database")
            saliency_score: str = Field(
                None, description="The score between 0 and 1")
        class SailencyContextList(BaseModel):
            """Buffer raw context processed by the buffer"""

            docs: List[SaliencyRawList] = Field(..., description="List of docs")
            observation: str = Field(..., description="The original user query")

        parser = PydanticOutputParser(pydantic_object=SailencyContextList)
        prompt = PromptTemplate(
            template="Determine saliency of documents compared to the other documents retrieved \n{format_instructions}\nOriginal observation is: {query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(query=observation)
        document_context_result = self.llm_base(_input.to_string())
        document_context_result_parsed = parser.parse(document_context_result)
        return document_context_result_parsed.json()



    async def handle_modulator(
        self,
        modulator_name: str,
        attention_modulators: Dict[str, float],
        observation: str,
        namespace: Optional[str] = None,
    ) -> Optional[List[Union[str, float]]]:
        """
        Handle the given modulator based on the observation and namespace.

        Parameters:
        - modulator_name: Name of the modulator to handle.
        - attention_modulators: Dictionary of modulator values.
        - observation: The current observation.
        - namespace: An optional namespace.

        Returns:
        - Result of the modulator if criteria met, else None.
        """
        modulator_value = attention_modulators.get(modulator_name, 0.0)
        modulator_functions = {
            "freshness": lambda obs, ns: self.freshness(observation=obs, namespace=ns),
            "frequency": lambda obs, ns: self.frequency(observation=obs, namespace=ns),
            "relevance": lambda obs, ns: self.relevance(observation=obs, namespace=ns),
            "saliency": lambda obs, ns: self.saliency(observation=obs, namespace=ns),
        }

        result_func = modulator_functions.get(modulator_name)
        if not result_func:
            return None

        result = await result_func(observation, namespace)
        if not result:
            return None

        try:
            if float(modulator_value) >= float(result[0]):
                return result
        except ValueError:
            pass

        return None

    async def available_operations(self) -> list[str]:
        """Determines what operations are available for the user to process PDFs"""

        return [
            "translate",
            "structure",
            "fetch from vector store"
            # "load to semantic memory",
            # "load to episodic memory",
            # "load to buffer",
        ]

    async def buffer_context(
        self,
        user_input=None,
        params=None,
        attention_modulators: dict = None,
    ):
        """Generates the context to be used for the buffer and passed to the agent"""
        try:
            # we delete all memories in the episodic buffer, so we can start fresh
            await self.delete_memories()
        except:
            # in case there are no memories, we pass
            pass
        # we just filter the data here to make sure input is clean
        prompt_filter = ChatPromptTemplate.from_template(
            "Filter and remove uneccessary information that is not relevant in the query to the vector store to get more information, keep it as original as possbile: {query}"
        )
        chain_filter = prompt_filter | self.llm
        output = await chain_filter.ainvoke({"query": user_input})

        # this part is unfinished but the idea is to apply different attention modulators to the data to fetch the most relevant information from the vector stores
        context = []
        if attention_modulators:
            print("HERE ARE THE ATTENTION MODULATORS: ", attention_modulators)
            from typing import Optional, Dict, List, Union

            lookup_value_semantic = await self.fetch_memories(
                observation=str(output), namespace="SEMANTICMEMORY"
            )
            context = []
            for memory in lookup_value_semantic["data"]["Get"]["SEMANTICMEMORY"]:
                # extract memory id, and pass it to fetch function as a parameter
                modulators = list(attention_modulators.keys())
                for modulator in modulators:
                    result = await self.handle_modulator(
                        modulator,
                        attention_modulators,
                        str(output),
                        namespace="EPISODICMEMORY",
                    )
                    if result:
                        context.append(result)
                        context.append(memory)
        else:
            # defaults to semantic search if we don't want to apply algorithms on the vectordb data
            lookup_value_episodic = await self.fetch_memories(
                observation=str(output), namespace="EPISODICMEMORY"
            )
            lookup_value_semantic = await self.fetch_memories(
                observation=str(output), namespace="SEMANTICMEMORY"
            )
            lookup_value_buffer = await self.fetch_memories(observation=str(output))

            context.append(lookup_value_buffer)
            context.append(lookup_value_semantic)
            context.append(lookup_value_episodic)

        class BufferModulators(BaseModel):
            frequency: str = Field(..., description="Frequency score of the document")
            saliency: str = Field(..., description="Saliency score of the document")
            relevance: str = Field(..., description="Relevance score of the document")

        class BufferRawContextTerms(BaseModel):
            """Schema for documentGroups"""

            semantic_search_term: str = Field(
                ...,
                description="The search term to use to get relevant input based on user query",
            )
            document_content: str = Field(
                None, description="Shortened original content of the document"
            )
            document_relevance: str = Field(
                None,
                description="The relevance of the document for the task on the scale from 0 to 1",
            )
            attention_modulators_list: List[BufferModulators] = Field(
                ..., description="List of modulators"
            )

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

        _input = prompt.format_prompt(query=user_input, context=context)
        document_context_result = self.llm_base(_input.to_string())
        document_context_result_parsed = parser.parse(document_context_result)
        print("HERE ARE THE DOCS PARSED AND STRUCTURED", document_context_result_parsed.json())

        return document_context_result_parsed

    async def get_task_list(
        self, user_input=None, params=None, attention_modulators=None,
    ):
        """Gets the task list from the document context result to enchance it and be able to pass to the agent"""
        list_of_operations = await self.available_operations()

        class Task(BaseModel):
            """Schema for an individual task."""

            task_order: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            task_name: str = Field(
                None, description="The task that needs to be performed"
            )
            operation: str = Field(None, description="The operation to be performed")
            original_query: str = Field(
                None, description="Original user query provided"
            )

        class TaskList(BaseModel):
            """Schema for the record containing a list of tasks."""

            tasks: List[Task] = Field(..., description="List of tasks")

        prompt_filter_chunk = f" Based on available operations {list_of_operations} determine only the relevant list of steps and operations sequentially based {user_input}"
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for decomposing prompts into steps and operations and choosing relevant ones"
            ),
            HumanMessage(content="Decompose based on the following prompt and provide relevant document context reponse:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
            HumanMessage(
                content="Tips: Only choose actions that are relevant to the user query and ignore others"
            )
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(
            TaskList, self.llm, prompt_, verbose=True
        )
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
        return tasks_list

    async def main_buffer(
        self, user_input=None, params=None, attention_modulators=None
    ):
        """AI buffer to run the AI agent to execute the set of tasks"""

        document_context_result_parsed = await self.buffer_context(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )
        tasks_list = await self.get_task_list(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators
        )
        result_tasks = []
        document_context_result_parsed = document_context_result_parsed.dict()
        document_from_vectorstore = [doc["document_content"] for doc in document_context_result_parsed["docs"]]

        for task in tasks_list:
            print("HERE IS THE TASK", task)

            complete_agent_prompt= f" Document context is: {document_from_vectorstore} \n  Task is : {task['task_order']} {task['task_name']} {task['operation']} "

            # task['vector_store_context_results']=document_context_result_parsed.dict()
            class PromptWrapper(BaseModel):
                observation: str = Field(
                    description="observation we want to fetch from vectordb"
                )

            @tool(
                "convert_to_structured", args_schema=PromptWrapper, return_direct=True
            )
            def convert_to_structured(observation=None, json_schema=None):
                """Convert unstructured data to structured data"""
                BASE_DIR = os.getcwd()
                json_path = os.path.join(
                    BASE_DIR, "schema_registry", "ticket_schema.json"
                )

                def load_json_or_infer_schema(file_path, document_path):
                    """Load JSON schema from file or infer schema from text"""

                    # Attempt to load the JSON file
                    with open(file_path, "r") as file:
                        json_schema = json.load(file)
                    return json_schema

                json_schema = load_json_or_infer_schema(json_path, None)

                def run_open_ai_mapper(observation=None, json_schema=None):
                    """Convert unstructured data to structured data"""

                    prompt_msgs = [
                        SystemMessage(
                            content="You are a world class algorithm converting unstructured data into structured data."
                        ),
                        HumanMessage(
                            content="Convert unstructured data to structured data:"
                        ),
                        HumanMessagePromptTemplate.from_template("{input}"),
                        HumanMessage(
                            content="Tips: Make sure to answer in the correct format"
                        ),
                    ]
                    prompt_ = ChatPromptTemplate(messages=prompt_msgs)
                    chain_funct = create_structured_output_chain(
                        json_schema, prompt=prompt_, llm=self.llm, verbose=True
                    )
                    output = chain_funct.run(input=observation, llm=self.llm)
                    return output

                result = run_open_ai_mapper(observation, json_schema)
                return result

            class FetchText(BaseModel):
                observation: str = Field(description="observation we want to translate")
            @tool("fetch_from_vector_store", args_schema=FetchText, return_direct=True)
            def fetch_from_vector_store(observation, args_schema=FetchText):
                """Fetch from vectorstore if data doesn't exist in the context"""
                if  document_context_result_parsed:
                    return document_context_result_parsed
                else:
                    out = self.fetch_memories(observation['original_query'], namespace="SEMANTICMEMORY")
                    return out

            class TranslateText(BaseModel):
                observation: str = Field(description="observation we want to translate")

            @tool("translate_to_de", args_schema=TranslateText, return_direct=True)
            def translate_to_de(observation, args_schema=TranslateText):
                """Translate to English"""
                out = GoogleTranslator(source="auto", target="de").translate(
                    text=observation
                )
                return out

            agent = initialize_agent(
                llm=self.llm,
                tools=[fetch_from_vector_store,translate_to_de, convert_to_structured],
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
            )

            output = agent.run(input=complete_agent_prompt )

            result_tasks.append(task)
            result_tasks.append(output)

        # print("HERE IS THE RESULT TASKS", str(result_tasks))
        #
        # buffer_result = await self.fetch_memories(observation=str(user_input))
        #
        # print("HERE IS THE RESULT TASKS", str(buffer_result))

        class EpisodicTask(BaseModel):
            """Schema for an individual task."""

            task_order: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            task_name: str = Field(
                None, description="The task that needs to be performed"
            )
            operation: str = Field(None, description="The operation to be performed")
            operation_result: str = Field(
                None, description="The result of the operation"
            )

        class EpisodicList(BaseModel):
            """Schema for the record containing a list of tasks."""

            tasks: List[EpisodicTask] = Field(..., description="List of tasks")
            start_date: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            end_date: str = Field(
                ..., description="The order at which the task needs to be performed"
            )
            user_query: str = Field(
                ..., description="The order at which the task needs to be performed"
            )

        parser = PydanticOutputParser(pydantic_object=EpisodicList)

        prompt = PromptTemplate(
            template="Format the result.\n{format_instructions}\nOriginal query is: {query}\n Steps are: {steps}, buffer is: {buffer}",
            input_variables=["query", "steps", "buffer"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        _input = prompt.format_prompt(
            query=user_input, steps=str(tasks_list)
            , buffer=str(result_tasks)
        )

        # return "a few things to do like load episodic memory in a structured format"
        output = self.llm_base(_input.to_string())
        result_parsing = parser.parse(output)
        lookup_value = await self.add_memories(
            observation=str(result_parsing.json()), params=params, namespace='EPISODICMEMORY'
        )
        # print("THE RESULT OF THIS QUERY IS ", result_parsing.json())
        await self.delete_memories()
        return result_parsing.json()


class LongTermMemory:
    def __init__(
        self,
        user_id: str = "676",
        memory_id: Optional[str] = None,
        index_name: Optional[str] = None,
        db_type: str = "weaviate",
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.ltm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.db_type = db_type
        self.semantic_memory = SemanticMemory(user_id, memory_id, index_name, db_type)
        self.episodic_memory = EpisodicMemory(user_id, memory_id, index_name, db_type)


class ShortTermMemory:
    def __init__(
        self,
        user_id: str = "676",
        memory_id: Optional[str] = None,
        index_name: Optional[str] = None,
        db_type: str = "weaviate",
    ):
        self.user_id = user_id
        self.memory_id = memory_id
        self.stm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.db_type = db_type
        self.episodic_buffer = EpisodicBuffer(user_id, memory_id, index_name, db_type)


class Memory:
    load_dotenv()
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    def __init__(
        self,
        user_id: str = "676",
        index_name: str = None,
        knowledge_source: str = None,
        knowledge_type: str = None,
        db_type: str = "weaviate",
        namespace: str = None,
    ) -> None:
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
    async def async_create_long_term_memory(
        self, user_id, memory_id, index_name, db_type
    ):
        # Perform asynchronous initialization steps if needed
        return LongTermMemory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )

    async def async_init(self):
        # Asynchronous initialization of LongTermMemory and ShortTermMemory
        self.long_term_memory = await self.async_create_long_term_memory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )
        self.short_term_memory = await self.async_create_short_term_memory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )

    async def async_create_short_term_memory(
        self, user_id, memory_id, index_name, db_type
    ):
        # Perform asynchronous initialization steps if needed
        return ShortTermMemory(
            user_id=self.user_id,
            memory_id=self.memory_id,
            index_name=self.index_name,
            db_type=self.db_type,
        )

    async def _add_semantic_memory(
        self, observation: str, loader_settings: dict = None, params: dict = None
    ):
        return await self.long_term_memory.semantic_memory.add_memories(
            observation=observation,
            loader_settings=loader_settings,
            params=params,
        )

    async def _fetch_semantic_memory(self, observation, params):
        return await self.long_term_memory.semantic_memory.fetch_memories(
            observation=observation, params=params
        )

    async def _delete_semantic_memory(self, params: str = None):
        return await self.long_term_memory.semantic_memory.delete_memories(
            params=params
        )

    async def _add_episodic_memory(
        self, observation: str, loader_settings: dict = None, params: dict = None
    ):
        return await self.long_term_memory.episodic_memory.add_memories(
            observation=observation, loader_settings=loader_settings, params=params
        )

    async def _fetch_episodic_memory(self, observation, params: str = None):
        return await self.long_term_memory.episodic_memory.fetch_memories(
            observation=observation, params=params
        )

    async def _delete_episodic_memory(self, params: str = None):
        return await self.long_term_memory.episodic_memory.delete_memories(
            params=params
        )


    async def _add_buffer_memory(
        self,
        user_input: str,
        namespace: str = None,
        loader_settings: dict = None,
        params: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.add_memories(
            observation=user_input, loader_settings=loader_settings, params=params
        )

    async def _fetch_buffer_memory(self, user_input: str):
        return await self.short_term_memory.episodic_buffer.fetch_memories(
            observation=user_input
        )

    async def _delete_buffer_memory(self, params: str = None):
        return await self.short_term_memory.episodic_buffer.delete_memories(
            params=params
        )

    async def _create_buffer_context(
        self,
        user_input: str,
        params: dict = None,
        attention_modulators: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.buffer_context(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )
    async def _get_task_list(
        self,
        user_input: str,
        params: str = None,
        attention_modulators: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.get_task_list(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )
    async def _run_main_buffer(
        self,
        user_input: str,
        params: dict = None,
        attention_modulators: dict = None,
    ):
        return await self.short_term_memory.episodic_buffer.main_buffer(
            user_input=user_input,
            params=params,
            attention_modulators=attention_modulators,
        )

    async def _available_operations(self):
        return await self.long_term_memory.episodic_buffer.available_operations()


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
        "validity_end": "2024-07-31",
    }

    # gg = await memory._run_buffer(user_input="i NEED TRANSLATION TO GERMAN ", content="i NEED TRANSLATION TO GERMAN ", params=params)
    # print(gg)

    # gg = await memory._fetch_buffer_memory(user_input="i  TO GERMAN ")
    # print(gg)


    modulator = {"relevance": 0.0, "saliency": 0.0, "frequency": 0.0}
    # #
    ggur = await memory._run_main_buffer(
        user_input="I want to know how does Buck adapt to life in the wild and then have that info translated to german ",
        params=params,
        attention_modulators=modulator,
    )
    print(ggur)

    ll =  {
    "format": "PDF",
    "source": "url",
    "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    }
    # ggur = await memory._add_semantic_memory(observation = "bla", loader_settings=ll, params=params)
    # print(ggur)
    # fff = await memory._delete_semantic_memory()
    # print(fff)

    # fff = await memory._fetch_semantic_memory(observation = "dog pulling sleds ", params=None)
    # print(fff)
    # print(len(fff["data"]["Get"]["EPISODICMEMORY"]))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

