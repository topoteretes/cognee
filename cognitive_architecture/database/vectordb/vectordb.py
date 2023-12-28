
# Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from marshmallow import Schema, fields
from cognitive_architecture.database.vectordb.loaders.loaders import _document_loader
# Add the parent directory to sys.path


logging.basicConfig(level=logging.INFO)
from langchain.retrievers import WeaviateHybridSearchRetriever, ParentDocumentRetriever
from weaviate.gql.get import HybridFusion
import tracemalloc
tracemalloc.start()
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document
import weaviate

load_dotenv()
from ...config import Config

config = Config()
config.load()

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
        namespace: str = None,
        embeddings = None,
    ):
        self.user_id = user_id
        self.index_name = index_name
        self.namespace = namespace
        self.memory_id = memory_id
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

    def init_weaviate(self,  embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY", "")), namespace=None,retriever_type="",):
        # Weaviate initialization logic
        auth_config = weaviate.auth.AuthApiKey(
            api_key=os.environ.get("WEAVIATE_API_KEY")
        )
        client = weaviate.Client(
            url=os.environ.get("WEAVIATE_URL"),
            auth_client_secret=auth_config,
            additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")},
        )

        if retriever_type == "single_document_context":
            retriever = WeaviateHybridSearchRetriever(
                client=client,
                index_name=namespace,
                text_key="text",
                attributes=[],
                embedding=embeddings,
                create_schema_if_missing=True,
            )
            return retriever
        elif retriever_type == "multi_document_context":
            retriever = WeaviateHybridSearchRetriever(
                client=client,
                index_name=namespace,
                text_key="text",
                attributes=[],
                embedding=embeddings,
                create_schema_if_missing=True,
            )
            return retriever
        else :
            return client
                # child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                # store = InMemoryStore()
                # retriever = ParentDocumentRetriever(
                #     vectorstore=vectorstore,
                #     docstore=store,
                #     child_splitter=child_splitter,
                # )
    from marshmallow import Schema, fields

    def create_document_structure(observation, params, metadata_schema_class=None):
        """
        Create and validate a document structure with optional custom fields.

        :param observation: Content of the document.
        :param params: Metadata information.
        :param metadata_schema_class: Custom metadata schema class (optional).
        :return: A list containing the validated document data.
        """
        document_data = {
            "metadata": params,
            "page_content": observation
        }

        def get_document_schema():
            class DynamicDocumentSchema(Schema):
                metadata = fields.Nested(metadata_schema_class, required=True)
                page_content = fields.Str(required=True)

            return DynamicDocumentSchema

        # Validate and deserialize, defaulting to "1.0" if not provided
        CurrentDocumentSchema = get_document_schema()
        loaded_document = CurrentDocumentSchema().load(document_data)
        return [loaded_document]

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
        params['user_id'] = self.user_id
        logging.info("User id is %s", self.user_id)
        retriever = self.init_weaviate(embeddings=OpenAIEmbeddings(),namespace = namespace, retriever_type="single_document_context")
        if loader_settings:
            # Assuming _document_loader returns a list of documents
            documents = await _document_loader(observation, loader_settings)
            logging.info("here are the docs %s", str(documents))
            chunk_count = 0
            for doc_list in documents:
                for doc in doc_list:
                    chunk_count += 1
                    params['chunk_count'] = doc.metadata.get("chunk_count", "None")
                    logging.info("Loading document with provided loader settings %s", str(doc))
                    params['source'] = doc.metadata.get("source", "None")
                    logging.info("Params are %s", str(params))
                    retriever.add_documents([
                Document(metadata=params, page_content=doc.page_content)])
        else:
            chunk_count = 0
            from cognitive_architecture.database.vectordb.chunkers.chunkers import chunk_data
            documents = [chunk_data(chunk_strategy="VANILLA", source_data=observation, chunk_size=300,
                       chunk_overlap=20)]
            for doc in documents[0]:
                chunk_count += 1
                params['chunk_order'] = chunk_count
                params['source'] = "User loaded"
                logging.info("Loading document with default loader settings %s", str(doc))
                logging.info("Params are %s", str(params))
                retriever.add_documents([
                Document(metadata=params, page_content=doc.page_content)])

    async def fetch_memories(self, observation: str, namespace: str = None, search_type: str = 'hybrid',params=None, **kwargs):
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
        client = self.init_weaviate(namespace =self.namespace)
        if search_type is None:
            search_type = 'hybrid'
        logging.info("The search type is s%", search_type)



        if not namespace:
            namespace = self.namespace

        logging.info("Query on namespace %s", namespace)

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

        n_of_observations = kwargs.get('n_of_observations', 2)

        # try:
        if search_type == 'text':
            query_output = (
                base_query
                .with_near_text({"concepts": [observation]})
                .with_autocut(n_of_observations)
                .do()
            )
        elif search_type == 'hybrid':
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
                .with_autocut(n_of_observations)
                .do()
            )
        elif search_type == 'summary':
            filter_object = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["user_id"],
                        "operator": "Equal",
                        "valueText": self.user_id,
                    },
                    {
                        "path": ["chunk_order"],
                        "operator": "LessThan",
                        "valueNumber": 30,
                    },
                ]
            }
            base_query = client.query.get(
                namespace, list(list_objects_of_class(namespace, client.schema.get()))
            ).with_additional(
                ["id", "creationTimeUnix", "lastUpdateTimeUnix", "score", 'distance']
            ).with_where(filter_object).with_limit(30)
            query_output = (
                base_query
                # .with_hybrid(query=observation, fusion_type=HybridFusion.RELATIVE_SCORE)
                .do()
            )

        elif search_type == 'summary_filter_by_object_name':
            filter_object = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["user_id"],
                        "operator": "Equal",
                        "valueText": self.user_id,
                    },
                    {
                        "path": ["doc_id"],
                        "operator": "Equal",
                        "valueText": params,
                    },
                ]
            }
            base_query = client.query.get(
                namespace, list(list_objects_of_class(namespace, client.schema.get()))
            ).with_additional(
                ["id", "creationTimeUnix", "lastUpdateTimeUnix", "score", 'distance']
            ).with_where(filter_object).with_limit(30).with_hybrid(query=observation, fusion_type=HybridFusion.RELATIVE_SCORE)
            query_output = (
                base_query
                .do()
            )
            # from weaviate.classes import Filter
            # client = weaviate.connect_to_wcs(
            #     cluster_url=config.weaviate_url,
            #     auth_credentials=weaviate.AuthApiKey(config.weaviate_api_key)
            # )

            return query_output
        elif search_type == 'generate':
            generate_prompt = kwargs.get('generate_prompt', "")
            query_output = (
                base_query
                .with_generate(single_prompt=observation)
                .with_near_text({"concepts": [observation]})
                .with_autocut(n_of_observations)
                .do()
            )
        elif search_type == 'generate_grouped':
            generate_prompt = kwargs.get('generate_prompt', "")
            query_output = (
                base_query
                .with_generate(grouped_task=observation)
                .with_near_text({"concepts": [observation]})
                .with_autocut(n_of_observations)
                .do()
            )
        else:
            logging.error(f"Invalid search_type: {search_type}")
            return []
        # except Exception as e:
        #     logging.error(f"Error executing query: {str(e)}")
        #     return []

        return query_output



    async def delete_memories(self, namespace:str, params: dict = None):
        if namespace is None:
            namespace = self.namespace
        client = self.init_weaviate(namespace = self.namespace)
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
            return client.batch.delete_objects(
                class_name=namespace,
                where={
                    "path": ["version"],
                    "operator": "Equal",
                    "valueText": "1.0",
                },
            )


    async def count_memories(self, namespace: str = None, params: dict = None) -> int:
        """
        Count memories in a Weaviate database.

        Args:
            namespace (str, optional): The Weaviate namespace to count memories in. If not provided, uses the default namespace.

        Returns:
            int: The number of memories in the specified namespace.
        """
        if namespace is None:
            namespace = self.namespace

        client = self.init_weaviate(namespace =namespace)

        try:
            object_count = client.query.aggregate(namespace).with_meta_count().do()
            return object_count
        except Exception as e:
            logging.info(f"Error counting memories: {str(e)}")
            # Handle the error or log it
            return 0

    def update_memories(self, observation, namespace: str, params: dict = None):
        client = self.init_weaviate(namespace = self.namespace)

        client.data_object.update(
            data_object={
                # "text": observation,
                "user_id": str(self.user_id),
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
