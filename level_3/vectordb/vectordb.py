
# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import logging

from marshmallow import Schema, fields
from loaders.loaders import _document_loader
# Add the parent directory to sys.path


logging.basicConfig(level=logging.INFO)
from langchain.retrievers import WeaviateHybridSearchRetriever
from weaviate.gql.get import HybridFusion
import tracemalloc
tracemalloc.start()
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
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
        embeddings = None,
    ):
        self.user_id = user_id
        self.index_name = index_name
        self.namespace = namespace
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.st_memory_id = st_memory_id
        self.buffer_id = buffer_id
        self.embeddings = embeddings

class PineconeVectorDB(VectorDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_pinecone(self.index_name)

    def init_pinecone(self, index_name):
        # Pinecone initialization logic
        pass

import langchain.embeddings
class WeaviateVectorDB(VectorDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_weaviate(embeddings= self.embeddings, namespace = self.namespace)

    def init_weaviate(self, embeddings =OpenAIEmbeddings() , namespace: str=None):
        # Weaviate initialization logic
        # embeddings = OpenAIEmbeddings()
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

    def _stuct(self, observation, params, metadata_schema_class =None):
        """Utility function to create the document structure with optional custom fields."""


        # Construct document data
        document_data = {
            "metadata": params,
            "page_content": observation
        }
        def get_document_schema():
            class DynamicDocumentSchema(Schema):
                metadata = fields.Nested(metadata_schema_class, required=True)
                page_content = fields.Str(required=True)

            return DynamicDocumentSchema
        # Validate and deserialize  # Default to "1.0" if not provided
        CurrentDocumentSchema = get_document_schema()
        loaded_document = CurrentDocumentSchema().load(document_data)
        return [loaded_document]
    async def add_memories(self, observation, loader_settings=None, params=None, namespace=None, metadata_schema_class=None, embeddings = 'hybrid'):
        # Update Weaviate memories here
        if namespace is None:
            namespace = self.namespace
        retriever = self.init_weaviate(embeddings=embeddings,namespace = namespace)
        if loader_settings:
            # Assuming _document_loader returns a list of documents
            documents = _document_loader(observation, loader_settings)
            logging.info("here are the docs %s", str(documents))
            for doc in documents:
                document_to_load = self._stuct(doc.page_content, params, metadata_schema_class)
                print("here is the doc to load1", document_to_load)
                retriever.add_documents([
            Document(metadata=document_to_load[0]['metadata'], page_content=document_to_load[0]['page_content'])])
        else:
            document_to_load = self._stuct(observation, params, metadata_schema_class)

            print("here is the doc to load2", document_to_load)
            retriever.add_documents([
            Document(metadata=document_to_load[0]['metadata'], page_content=document_to_load[0]['page_content'])])

    async def fetch_memories(self, observation: str, namespace: str = None, search_type: str = 'hybrid', **kwargs):
        """
        Fetch documents from weaviate.

        Parameters:
        - observation (str): User query.
        - namespace (str, optional): Type of memory accessed.
        - search_type (str, optional): Type of search ('text', 'hybrid', 'bm25', 'generate', 'generate_grouped'). Defaults to 'hybrid'.
        - **kwargs: Additional parameters for flexibility.

        Returns:
        List of documents matching the query or an empty list in case of error.

        Example:
            fetch_memories(query="some query", search_type='text', additional_param='value')
        """
        client = self.init_weaviate_client(self.namespace)
        if search_type is None:
            search_type = 'hybrid'


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

        try:
            if search_type == 'text':
                query_output = (
                    base_query
                    .with_near_text({"concepts": [observation]})
                    .do()
                )
            elif search_type == 'hybrid':
                n_of_observations = kwargs.get('n_of_observations', 2)


                query_output = (
                    base_query
                    .with_hybrid(query=observation, fusion_type=HybridFusion.RELATIVE_SCORE)
                    .with_autocut(n_of_observations)
                    .do()
                )
            elif search_type == 'bm25':
                query_output = (
                    base_query
                    .with_bm25(query=observation)
                    .do()
                )
            elif search_type == 'generate':
                generate_prompt = kwargs.get('generate_prompt', "")
                query_output = (
                    base_query
                    .with_generate(single_prompt=generate_prompt)
                    .with_near_text({"concepts": [observation]})
                    .do()
                )
            elif search_type == 'generate_grouped':
                generate_prompt = kwargs.get('generate_prompt', "")
                query_output = (
                    base_query
                    .with_generate(grouped_task=generate_prompt)
                    .with_near_text({"concepts": [observation]})
                    .do()
                )
            else:
                logging.error(f"Invalid search_type: {search_type}")
                return []
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            return []

        return query_output

    async def delete_memories(self, namespace:str, params: dict = None):
        if namespace is None:
            namespace = self.namespace
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
                class_name=namespace,
                where={
                    "path": ["version"],
                    "operator": "Equal",
                    "valueText": "1.0",
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
