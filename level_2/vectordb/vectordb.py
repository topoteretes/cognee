
# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import logging
from io import BytesIO

import sys
import os

from marshmallow import Schema, fields
from level_2.loaders.loaders import _document_loader
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
import marvin
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import WeaviateHybridSearchRetriever
from weaviate.gql.get import HybridFusion
import tracemalloc
tracemalloc.start()
import os
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from level_2.schema.semantic.semantic_schema import DocumentSchema, SCHEMA_VERSIONS, DocumentMetadataSchemaV1
from langchain.schema import Document
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

    # def _document_loader(self, observation: str, loader_settings: dict):
    #     # Check the format of the document
    #     document_format = loader_settings.get("format", "text")
    #
    #     if document_format == "PDF":
    #         if loader_settings.get("source") == "url":
    #             pdf_response = requests.get(loader_settings["path"])
    #             pdf_stream = BytesIO(pdf_response.content)
    #             contents = pdf_stream.read()
    #             tmp_location = os.path.join("/tmp", "tmp.pdf")
    #             with open(tmp_location, "wb") as tmp_file:
    #                 tmp_file.write(contents)
    #
    #             # Process the PDF using PyPDFLoader
    #             loader = PyPDFLoader(tmp_location)
    #             # adapt this for different chunking strategies
    #             pages = loader.load_and_split()
    #             return pages
    #         elif loader_settings.get("source") == "file":
    #             # Process the PDF using PyPDFLoader
    #             # might need adapting for different loaders + OCR
    #             # need to test the path
    #             loader = PyPDFLoader(loader_settings["path"])
    #             pages = loader.load_and_split()
    #             return pages
    #
    #     elif document_format == "text":
    #         # Process the text directly
    #         return observation
    #
    #     else:
    #         raise ValueError(f"Unsupported document format: {document_format}")
    def _stuct(self, observation, params, custom_fields=None):
        """Utility function to create the document structure with optional custom fields."""
        # Dynamically construct metadata
        metadata = {
            key: str(getattr(self, key, params.get(key, "")))
            for key in [
                "user_id", "memory_id", "ltm_memory_id",
                "st_memory_id", "buffer_id", "version",
                "agreement_id", "privacy_policy", "terms_of_service",
                "format", "schema_version", "checksum",
                "owner", "license", "validity_start", "validity_end"
            ]
        }
        # Merge with custom fields if provided
        if custom_fields:
            metadata.update(custom_fields)

        # Construct document data
        document_data = {
            "metadata": metadata,
            "page_content": observation
        }

        def get_document_schema_based_on_version(version):
            metadata_schema_class = SCHEMA_VERSIONS.get(version, DocumentMetadataSchemaV1)
            class DynamicDocumentSchema(Schema):
                metadata = fields.Nested(metadata_schema_class, required=True)
                page_content = fields.Str(required=True)

            return DynamicDocumentSchema

        # Validate and deserialize
        schema_version = params.get("schema_version", "1.0")  # Default to "1.0" if not provided
        CurrentDocumentSchema = get_document_schema_based_on_version(schema_version)
        loaded_document = CurrentDocumentSchema().load(document_data)
        return [loaded_document]

    async def add_memories(self, observation, loader_settings=None, params=None, namespace=None, custom_fields=None):
        # Update Weaviate memories here
        if namespace is None:
            namespace = self.namespace
        retriever = self.init_weaviate(namespace)  # Assuming `init_weaviate` is a method of the class
        if loader_settings:
            # Assuming _document_loader returns a list of documents
            documents = _document_loader(observation, loader_settings)
            for doc in documents:
                document_to_load = self._stuct(doc.page_content, params, custom_fields)
                print("here is the doc to load1", document_to_load)
                retriever.add_documents([
            Document(metadata=document_to_load[0]['metadata'], page_content=document_to_load[0]['page_content'])])
        else:
            document_to_load = self._stuct(observation, params, custom_fields)
            retriever.add_documents([
            Document(metadata=document_to_load[0]['metadata'], page_content=document_to_load[0]['page_content'])])

    async def fetch_memories(
            self, observation: str, namespace: str, params: dict = None, n_of_observations: int = 2
    ):
        """
        Fetch documents from weaviate.

        Parameters:
        - observation (str): User query.
        - namespace (str): Type of memory accessed.
        - params (dict, optional): Filtering parameters.
        - n_of_observations (int, optional): For weaviate, equals to autocut. Defaults to 2. Ranges from 1 to 3.

        Returns:
        List of documents matching the query.

        Example:
            fetch_memories(query="some query", path=['year'], operator='Equal', valueText='2017*')
        """
        client = self.init_weaviate_client(self.namespace)

        if not namespace:
            namespace = self.namespace

        params_user_id = {
            "path": ["user_id"],
            "operator": "Like",
            "valueText": self.user_id,
        }

        def list_objects_of_class(class_name, schema):
            return [
                prop["name"]
                for class_obj in schema["classes"]
                if class_obj["class"] == class_name
                for prop in class_obj["properties"]
            ]

        base_query = client.query.get(
            namespace, list(list_objects_of_class(namespace, client.schema.get()))
        ).with_additional(
            ["id", "creationTimeUnix", "lastUpdateTimeUnix", "score", 'distance']
        ).with_where(params_user_id).with_limit(10)

        if params:
            query_output = (
                base_query
                .with_where(params)
                .with_near_text({"concepts": [observation]})
                .do()
            )
        else:
            query_output = (
                base_query
                .with_hybrid(
                    query=observation,
                    fusion_type=HybridFusion.RELATIVE_SCORE
                )
                .with_autocut(n_of_observations)
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
