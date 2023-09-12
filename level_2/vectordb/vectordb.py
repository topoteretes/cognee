
# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import logging
from io import BytesIO



logging.basicConfig(level=logging.INFO)
import marvin
import requests
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import WeaviateHybridSearchRetriever
from weaviate.gql.get import HybridFusion

load_dotenv()
from typing import Optional

import tracemalloc

tracemalloc.start()

import os
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document
import uuid
import weaviate

load_dotenv()


LTM_MEMORY_ID_DEFAULT = "00000"
ST_MEMORY_ID_DEFAULT = "0000"
BUFFER_ID_DEFAULT = "0000"
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
        # Check the format of the document
        document_format = loader_settings.get("format", "text")

        if document_format == "PDF":
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
            elif loader_settings.get("source") == "file":
                # Process the PDF using PyPDFLoader
                # might need adapting for different loaders + OCR
                # need to test the path
                loader = PyPDFLoader(loader_settings["path"])
                pages = loader.load_and_split()
                return pages

        elif document_format == "text":
            # Process the text directly
            return observation

        else:
            raise ValueError(f"Unsupported document format: {document_format}")

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
        self, observation: str, namespace: str, params: dict = None, n_of_observations =int(2)
    ):
        """
        Get documents from weaviate.

            Parameters:
            - observation (str): User query.
            - namespace (str): Type of memory we access.
            - params (dict, optional):
            - n_of_observations (int, optional): For weaviate, equals to autocut, defaults to 1. Ranges from 1 to 3. Check weaviate docs for more info.

            Returns:
            Describe the return type and what the function returns.

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
                .with_limit(10)
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
                    fusion_type=HybridFusion.RELATIVE_SCORE
                )
                .with_autocut(n_of_observations)
                .with_where(params_user_id)
                .with_limit(10)
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
