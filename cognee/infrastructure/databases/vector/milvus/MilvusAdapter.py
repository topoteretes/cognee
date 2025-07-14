from __future__ import annotations
import asyncio
import os
from uuid import UUID
from typing import List, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.files.storage import get_file_storage

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..vector_db_interface import VectorDBInterface

logger = get_logger("MilvusAdapter")


class IndexSchema(DataPoint):
    """
    Represent a schema for an index that includes text data and associated metadata.

    This class inherits from DataPoint and includes attributes for text and metadata. It
    defines the structure of the data points used in the index, holding the text as a string
    and metadata as a dictionary with predefined index fields.
    """

    text: str

    metadata: dict = {"index_fields": ["text"]}


class MilvusAdapter(VectorDBInterface):
    """
    Interface for interacting with a Milvus vector database.

    Public methods:

    - __init__
    - get_milvus_client
    - embed_data
    - has_collection
    - create_collection
    - create_data_points
    - create_vector_index
    - index_data_points
    - retrieve
    - search
    - batch_search
    - delete_data_points
    - prune
    """

    name = "Milvus"
    url: str
    api_key: Optional[str]
    embedding_engine: EmbeddingEngine = None

    def __init__(self, url: str, api_key: Optional[str], embedding_engine: EmbeddingEngine):
        self.url = url
        self.api_key = api_key

        self.embedding_engine = embedding_engine

    def get_milvus_client(self):
        """
        Retrieve a Milvus client instance.

        Returns a MilvusClient object configured with the provided URL and optional API key.

        Returns:
        --------

            A MilvusClient instance.
        """
        from pymilvus import MilvusClient

        # Ensure the parent directory exists for local file-based Milvus databases
        if self.url and not self.url.startswith(("http://", "https://", "grpc://")):
            # This is likely a local file path, ensure the directory exists
            db_dir = os.path.dirname(self.url)
            if db_dir and not os.path.exists(db_dir):
                try:
                    file_storage = get_file_storage(db_dir)
                    if hasattr(file_storage, "ensure_directory_exists"):
                        if asyncio.iscoroutinefunction(file_storage.ensure_directory_exists):
                            # Run async function synchronously in this sync method
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # If we're already in an async context, we can't use run_sync easily
                                # Create the directory directly as a fallback
                                os.makedirs(db_dir, exist_ok=True)
                            else:
                                loop.run_until_complete(file_storage.ensure_directory_exists())
                        else:
                            file_storage.ensure_directory_exists()
                    else:
                        # Fallback to os.makedirs if file_storage doesn't have ensure_directory_exists
                        os.makedirs(db_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(
                        f"Could not create directory {db_dir} using file_storage, falling back to os.makedirs: {e}"
                    )
                    os.makedirs(db_dir, exist_ok=True)

        if self.api_key:
            client = MilvusClient(uri=self.url, token=self.api_key)
        else:
            client = MilvusClient(uri=self.url)
        return client

    async def embed_data(self, data: List[str]) -> list[list[float]]:
        """
        Embed a list of text data into vectors asynchronously.

        Accepts a list of strings and utilizes the embedding engine to convert them into
        vectors.

        Parameters:
        -----------

            - data (List[str]): A list of textual data to be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists containing embedded vectors.
        """
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the database asynchronously.

        Returns a boolean indicating whether the specified collection is present.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for its existence.

        Returns:
        --------

            - bool: True if the collection exists, False otherwise.
        """
        future = asyncio.Future()
        client = self.get_milvus_client()
        future.set_result(client.has_collection(collection_name=collection_name))

        return await future

    async def create_collection(
        self,
        collection_name: str,
        payload_schema=None,
    ):
        """
        Create a new collection in the vector database asynchronously.

        Raises a MilvusException if there are issues creating the collection, such as already
        existing collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to be created.
            - payload_schema: Optional schema for the collection, defaults to None if not
              provided. (default None)

        Returns:
        --------

            True if the collection is created successfully, otherwise returns None.
        """
        from pymilvus import DataType, MilvusException

        client = self.get_milvus_client()
        if client.has_collection(collection_name=collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            return True

        try:
            dimension = self.embedding_engine.get_vector_size()
            assert dimension > 0, "Embedding dimension must be greater than 0."

            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )

            schema.add_field(
                field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=36
            )

            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=60535)

            index_params = client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="COSINE")

            client.create_collection(
                collection_name=collection_name, schema=schema, index_params=index_params
            )

            client.load_collection(collection_name)

            logger.info(f"Collection '{collection_name}' created successfully.")
            return True
        except MilvusException as e:
            logger.error(f"Error creating collection '{collection_name}': {str(e)}")
            raise e

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """
        Insert multiple data points into a specified collection asynchronously.

        Raises CollectionNotFoundError if the specified collection does not exist.

        Parameters:
        -----------

            - collection_name (str): The name of the collection where data points will be
              inserted.
            - data_points (List[DataPoint]): A list of DataPoint objects to be inserted into the
              collection.

        Returns:
        --------

            The result of the insert operation, includes count of inserted data points.
        """
        from pymilvus import MilvusException, exceptions

        client = self.get_milvus_client()
        data_vectors = await self.embed_data(
            [data_point.get_embeddable_data(data_point) for data_point in data_points]
        )

        insert_data = [
            {
                "id": str(data_point.id),
                "vector": data_vectors[index],
                "text": data_point.text,
            }
            for index, data_point in enumerate(data_points)
        ]

        try:
            result = client.insert(collection_name=collection_name, data=insert_data)
            logger.info(
                f"Inserted {result.get('insert_count', 0)} data points into collection '{collection_name}'."
            )
            return result
        except exceptions.CollectionNotExistException as error:
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' does not exist!"
            ) from error
        except MilvusException as e:
            logger.error(
                f"Error inserting data points into collection '{collection_name}': {str(e)}"
            )
            raise e

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """
        Create a vector index for a given collection asynchronously.

        Parameters:
        -----------

            - index_name (str): The name of the vector index being created.
            - index_property_name (str): The property name associated with the index.
        """
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        """
        Index the provided data points into the collection based on index names asynchronously.

        Parameters:
        -----------

            - index_name (str): The name of the index where data points will be indexed.
            - index_property_name (str): The property name associated with the index.
            - data_points (List[DataPoint]): A list of DataPoint objects to be indexed.
        """
        formatted_data_points = [
            IndexSchema(
                id=data_point.id,
                text=getattr(data_point, data_point.metadata["index_fields"][0]),
            )
            for data_point in data_points
        ]
        collection_name = f"{index_name}_{index_property_name}"
        await self.create_data_points(collection_name, formatted_data_points)

    async def retrieve(self, collection_name: str, data_point_ids: list[UUID]):
        """
        Retrieve data points from a collection based on their IDs asynchronously.

        Raises CollectionNotFoundError if the specified collection does not exist.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which data points will be
              retrieved.
            - data_point_ids (list[UUID]): A list of UUIDs representing the IDs of the data
              points to be retrieved.

        Returns:
        --------

            The results of the query, including the requested data points.
        """
        from pymilvus import MilvusException, exceptions

        client = self.get_milvus_client()
        try:
            filter_expression = f"""id in [{", ".join(f'"{id}"' for id in data_point_ids)}]"""

            results = client.query(
                collection_name=collection_name,
                expr=filter_expression,
                output_fields=["*"],
            )
            return results
        except exceptions.CollectionNotExistException as error:
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' does not exist!"
            ) from error
        except MilvusException as e:
            logger.error(
                f"Error retrieving data points from collection '{collection_name}': {str(e)}"
            )
            raise e

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        """
        Search for data points in a collection based on a text query or vector asynchronously.

        Raises ValueError if neither query_text nor query_vector is provided. Raises
        MilvusException for errors during the search process.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to search within.
            - query_text (Optional[str]): Optional text query used for searching, defaults to
              None. (default None)
            - query_vector (Optional[List[float]]): Optional vector query used for searching,
              defaults to None. (default None)
            - limit (int): Maximum number of results to return, defaults to 15. (default 15)
            - with_vector (bool): Flag to indicate if the vector should be included in the
              results, defaults to False. (default False)

        Returns:
        --------

            A list of scored results that match the query; may include vector data if requested.
        """
        from pymilvus import MilvusException, exceptions

        client = self.get_milvus_client()
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        if not client.has_collection(collection_name=collection_name):
            logger.warning(
                f"Collection '{collection_name}' not found in MilvusAdapter.search; returning []."
            )
            return []

        try:
            query_vector = query_vector or (await self.embed_data([query_text]))[0]

            output_fields = ["id", "text"]
            if with_vector:
                output_fields.append("vector")

            results = client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=limit if limit > 0 else None,
                output_fields=output_fields,
                search_params={
                    "metric_type": "COSINE",
                },
            )

            return [
                ScoredResult(
                    id=parse_id(result["id"]),
                    score=result["distance"],
                    payload=result.get("entity", {}),
                )
                for result in results[0]
            ]
        except exceptions.CollectionNotExistException:
            logger.warning(
                f"Collection '{collection_name}' not found (exception) in MilvusAdapter.search; returning []."
            )
            return []
        except MilvusException as e:
            # Catch other Milvus errors that are "collection not found" (paranoid safety)
            if "collection not found" in str(e).lower() or "schema" in str(e).lower():
                logger.warning(
                    f"Collection '{collection_name}' not found (MilvusException) in MilvusAdapter.search; returning []."
                )
                return []
            logger.error(f"Error searching Milvus collection '{collection_name}': {e}")
            raise e

    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        """
        Perform a batch search in a collection for multiple textual queries asynchronously.

        Utilizes embed_data to convert texts into vectors and returns the search results for
        each query.

        Parameters:
        -----------

            - collection_name (str): The name of the collection where the search will be
              performed.
            - query_texts (List[str]): A list of texts to search for in the collection.
            - limit (int): Maximum number of results to return per query.
            - with_vectors (bool): Specifies if the vectors should be included in the search
              results, defaults to False. (default False)

        Returns:
        --------

            A list of search result sets, one for each query input.
        """
        query_vectors = await self.embed_data(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        """
        Delete specific data points from a collection based on their IDs asynchronously.

        Raises MilvusException for errors during the deletion process.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which data points will be
              deleted.
            - data_point_ids (list[UUID]): A list of UUIDs representing the IDs of the data
              points to be deleted.

        Returns:
        --------

            The result of the delete operation, indicating success or failure.
        """
        from pymilvus import MilvusException

        client = self.get_milvus_client()
        try:
            filter_expression = f"""id in [{", ".join(f'"{id}"' for id in data_point_ids)}]"""

            delete_result = client.delete(collection_name=collection_name, filter=filter_expression)

            logger.info(
                f"Deleted data points with IDs {data_point_ids} from collection '{collection_name}'."
            )
            return delete_result
        except MilvusException as e:
            logger.error(
                f"Error deleting data points from collection '{collection_name}': {str(e)}"
            )
            raise e

    async def prune(self):
        """
        Remove all collections from the connected Milvus client asynchronously.
        """
        client = self.get_milvus_client()
        if client:
            collections = client.list_collections()
            for collection_name in collections:
                client.drop_collection(collection_name=collection_name)
            client.close()
