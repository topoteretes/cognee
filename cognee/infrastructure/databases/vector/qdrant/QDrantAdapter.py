import os
from typing import Dict, List, Optional
from qdrant_client import AsyncQdrantClient, models

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine.utils import parse_id
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..vector_db_interface import VectorDBInterface

logger = get_logger("QDrantAdapter")


class IndexSchema(DataPoint):
    """
    Represents a schema for indexing where each data point contains a text field.

    This class inherits from DataPoint and defines a text attribute as well as metadata
    containing index fields used for indexing operations.
    """

    text: str

    metadata: dict = {"index_fields": ["text"]}


# class CollectionConfig(BaseModel, extra = "forbid"):
#     vector_config: Dict[str, models.VectorParams] = Field(..., description="Vectors configuration" )
#     hnsw_config: Optional[models.HnswConfig] = Field(default = None, description="HNSW vector index configuration")
#     optimizers_config: Optional[models.OptimizersConfig] = Field(default = None, description="Optimizers configuration")
#     quantization_config: Optional[models.QuantizationConfig] = Field(default = None, description="Quantization configuration")


def create_hnsw_config(hnsw_config: Dict):
    """
    Create HNSW configuration.

    This function returns an HNSW configuration object if the provided configuration is not
    None, otherwise it returns None.

    Parameters:
    -----------

        - hnsw_config (Dict): A dictionary containing HNSW configuration parameters.

    Returns:
    --------

        An instance of models.HnswConfig if hnsw_config is not None, otherwise None.
    """
    if hnsw_config is not None:
        return models.HnswConfig()
    return None


def create_optimizers_config(optimizers_config: Dict):
    """
    Create and return an OptimizersConfig instance if the input configuration is provided.

    This function checks if the given optimizers configuration is not None. If valid, it
    initializes and returns a new instance of the OptimizersConfig class from the models
    module. If the configuration is None, it returns None instead.

    Parameters:
    -----------

        - optimizers_config (Dict): A dictionary containing optimizer configuration
          settings.

    Returns:
    --------

        Returns an instance of OptimizersConfig if optimizers_config is provided; otherwise,
        returns None.
    """
    if optimizers_config is not None:
        return models.OptimizersConfig()
    return None


def create_quantization_config(quantization_config: Dict):
    """
    Create a quantization configuration based on the provided settings.

    This function generates an instance of `QuantizationConfig` if the provided
    `quantization_config` is not None. If it is None, the function returns None.

    Parameters:
    -----------

        - quantization_config (Dict): A dictionary containing the quantization configuration
          settings.

    Returns:
    --------

        An instance of `QuantizationConfig` if `quantization_config` is provided; otherwise,
        returns None.
    """
    if quantization_config is not None:
        return models.QuantizationConfig()
    return None


class QDrantAdapter(VectorDBInterface):
    """
    Adapt to the Qdrant vector database interface.

    Public methods:
    - get_qdrant_client
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

    name = "Qdrant"
    url: str = None
    api_key: str = None
    qdrant_path: str = None

    def __init__(self, url, api_key, embedding_engine: EmbeddingEngine, qdrant_path=None):
        self.embedding_engine = embedding_engine

        if qdrant_path is not None:
            self.qdrant_path = qdrant_path
        else:
            self.url = url
            self.api_key = api_key

    def get_qdrant_client(self) -> AsyncQdrantClient:
        """
        Retrieve an instance of AsyncQdrantClient configured with the appropriate
        settings based on the instance's attributes.

        Returns an instance of AsyncQdrantClient configured to connect to the database.

        Returns:
        --------
            - AsyncQdrantClient: An instance of AsyncQdrantClient configured for database
              operations.
        """
        is_prod = os.getenv("ENV").lower() == "prod"

        if self.qdrant_path is not None:
            return AsyncQdrantClient(path=self.qdrant_path, port=6333, https=is_prod)
        elif self.url is not None:
            return AsyncQdrantClient(url=self.url, api_key=self.api_key, port=6333, https=is_prod)

        return AsyncQdrantClient(location=":memory:")

    async def embed_data(self, data: List[str]) -> List[float]:
        """
        Embed a list of text data into vector representations asynchronously.

        Parameters:
        -----------

            - data (List[str]): A list of strings containing the text data to be embedded.

        Returns:
        --------

            - List[float]: A list of floating-point vectors representing the embedded text data.
        """
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists in the Qdrant database asynchronously.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: True if the specified collection exists, False otherwise.
        """
        client = self.get_qdrant_client()
        result = await client.collection_exists(collection_name)
        await client.close()
        return result

    async def create_collection(
        self,
        collection_name: str,
        payload_schema=None,
    ):
        """
        Create a new collection in the Qdrant database if it does not already exist.

        If the collection already exists, this operation has no effect.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to create.
            - payload_schema: Optional schema for the payload. Defaults to None. (default None)
        """
        client = self.get_qdrant_client()

        if not await client.collection_exists(collection_name):
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=self.embedding_engine.get_vector_size(), distance="Cosine"
                    )
                },
            )

        await client.close()

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """
        Create and upload data points to a specified collection in the database.

        Raises CollectionNotFoundError if the collection does not exist.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to which data points will be
              uploaded.
            - data_points (List[DataPoint]): A list of DataPoint objects to be uploaded.

        Returns:
        --------

            None if the operation is successful; raises exceptions on error.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = self.get_qdrant_client()

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        def convert_to_qdrant_point(data_point: DataPoint):
            """
            Convert a DataPoint object into the format expected by Qdrant for upload.

            Parameters:
            -----------

                - data_point (DataPoint): The DataPoint object to convert.

            Returns:
            --------

                None; performs an operation without returning a value.
            """
            return models.PointStruct(
                id=str(data_point.id),
                payload=data_point.model_dump(),
                vector={"text": data_vectors[data_points.index(data_point)]},
            )

        points = [convert_to_qdrant_point(point) for point in data_points]

        try:
            client.upload_points(collection_name=collection_name, points=points)
        except UnexpectedResponse as error:
            if "Collection not found" in str(error):
                raise CollectionNotFoundError(
                    message=f"Collection {collection_name} not found!"
                ) from error
            else:
                raise error
        except Exception as error:
            logger.error("Error uploading data points to Qdrant: %s", str(error))
            raise error
        finally:
            await client.close()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """
        Create a vector index for a specified property name.

        This is essentially a wrapper around create_collection, which allows for more
        flexibility
        in index naming.

        Parameters:
        -----------

            - index_name (str): The base name for the index to be created.
            - index_property_name (str): The property name that will be part of the index name.
        """
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """
        Index data points into a specific collection based on provided metadata.

        Transforms DataPoint objects into an appropriate format and uploads them.

        Parameters:
        -----------

            - index_name (str): The base name for the index used for naming the collection.
            - index_property_name (str): The property name used for naming the collection.
            - data_points (list[DataPoint]): A list of DataPoint objects to index.
        """
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """
        Retrieve data points from a specified collection based on their IDs.

        Returns the data corresponding to the provided IDs from the collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to retrieve from.
            - data_point_ids (list[str]): A list of IDs of the data points to retrieve.

        Returns:
        --------

            The retrieved data points, including payloads for each ID.
        """
        client = self.get_qdrant_client()
        results = await client.retrieve(collection_name, data_point_ids, with_payload=True)
        await client.close()
        return results

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        """
        Search for data points in a collection based on either a textual query or a vector
        query.

        Raises InvalidValueError if both query_text and query_vector are None.

        Returns a list of scored results that match the search criteria.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to search within.
            - query_text (Optional[str]): The text to be used in the search query; optional if
              query_vector is provided. (default None)
            - query_vector (Optional[List[float]]): The vector to be used in the search query;
              optional if query_text is provided. (default None)
            - limit (int): The maximum number of results to return; defaults to 15. (default 15)
            - with_vector (bool): Indicates whether to return vector data along with results;
              defaults to False. (default False)

        Returns:
        --------

            - List[ScoredResult]: A list of ScoredResult objects representing the results of the
              search.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if not await self.has_collection(collection_name):
            return []

        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]

        try:
            client = self.get_qdrant_client()
            if limit == 0:
                collection_size = await client.count(collection_name=collection_name)

            results = await client.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(
                    name="text",
                    vector=query_vector
                    if query_vector is not None
                    else (await self.embed_data([query_text]))[0],
                ),
                limit=limit if limit > 0 else collection_size.count,
                with_vectors=with_vector,
            )

            await client.close()

            return [
                ScoredResult(
                    id=parse_id(result.id),
                    payload={
                        **result.payload,
                        "id": parse_id(result.id),
                    },
                    score=1 - result.score,
                )
                for result in results
            ]
        finally:
            await client.close()

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        """
        Perform a batch search in a specified collection using multiple query texts.

        Returns the results of the search for each query, filtering for results with a score
        higher than 0.9.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to search in.
            - query_texts (List[str]): A list of query texts to search for in the collection.
            - limit (int): The maximum number of results to return for each search request; can
              be None. (default None)
            - with_vectors (bool): Indicates whether to include vector data in the results;
              defaults to False. (default False)

        Returns:
        --------

            A list containing the filtered search results for each query text.
        """

        vectors = await self.embed_data(query_texts)

        # Generate dynamic search requests based on the provided embeddings
        requests = [
            models.SearchRequest(
                vector=models.NamedVector(name="text", vector=vector),
                limit=limit,
                with_vector=with_vectors,
            )
            for vector in vectors
        ]

        client = self.get_qdrant_client()

        # Perform batch search with the dynamically generated requests
        results = await client.search_batch(collection_name=collection_name, requests=requests)

        await client.close()

        return [filter(lambda result: result.score > 0.9, result_group) for result_group in results]

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        """
        Delete specific data points from a specified collection based on their IDs.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete the data
              points.
            - data_point_ids (list[str]): The list of IDs of data points to be deleted.

        Returns:
        --------

            The result of the delete operation from the database.
        """
        client = self.get_qdrant_client()
        results = await client.delete(collection_name, data_point_ids)
        return results

    async def prune(self):
        """
        Remove all collections from the Qdrant database asynchronously.
        """
        client = self.get_qdrant_client()

        response = await client.get_collections()

        for collection in response.collections:
            await client.delete_collection(collection.name)

        await client.close()
