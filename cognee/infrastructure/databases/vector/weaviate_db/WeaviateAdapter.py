from typing import List, Optional

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from cognee.shared.logging_utils import get_logger
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

from distributed.utils import override_distributed
from distributed.tasks.queued_add_data_points import queued_add_data_points

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..vector_db_interface import VectorDBInterface

logger = get_logger("WeaviateAdapter")


def is_retryable_request(error):
    from weaviate.exceptions import UnexpectedStatusCodeException
    from requests.exceptions import RequestException

    if isinstance(error, UnexpectedStatusCodeException):
        # Retry on conflict, service unavailable, internal error
        return error.status_code in {409, 503, 500}
    if isinstance(error, RequestException):
        return True  # Includes timeout, connection error, etc.
    return False


class IndexSchema(DataPoint):
    """
    Define a schema for indexing data points with textual content.

    The IndexSchema class inherits from DataPoint and includes the following public
    attributes:

    - text: A string representing the main content of the data point.
    - metadata: A dictionary containing indexing information, specifically the fields to be
    indexed (in this case, the 'text' field).
    """

    text: str

    metadata: dict = {"index_fields": ["text"]}


class WeaviateAdapter(VectorDBInterface):
    """
    Adapt the Weaviate vector database to an interface for managing collections and data
    points.

    Public methods:
    - get_client
    - embed_data
    - has_collection
    - create_collection
    - get_collection
    - create_data_points
    - create_vector_index
    - index_data_points
    - retrieve
    - search
    - batch_search
    - delete_data_points
    - prune
    """

    name = "Weaviate"
    url: str
    api_key: str
    embedding_engine: EmbeddingEngine = None

    def __init__(self, url: str, api_key: str, embedding_engine: EmbeddingEngine):
        import weaviate
        import weaviate.classes as wvc

        self.url = url
        self.api_key = api_key

        self.embedding_engine = embedding_engine

        self.client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=30)),
        )

    async def get_client(self):
        """
        Establish a connection to the Weaviate client.

        Return the Weaviate client instance after connecting asynchronously.

        Returns:
        --------

            The Weaviate client instance.
        """
        await self.client.connect()

        return self.client

    async def embed_data(self, data: List[str]) -> List[float]:
        """
        Embed the given text data into vector representations.

        Given a list of strings, return their vector embeddings using the configured embedding
        engine.

        Parameters:
        -----------

            - data (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - List[float]: A list of float vectors corresponding to the embedded text data.
        """
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the Weaviate database.

        Return a boolean indicating the presence of the specified collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check.

        Returns:
        --------

            - bool: True if the collection exists, otherwise False.
        """
        client = await self.get_client()
        return await client.collections.exists(collection_name)

    @retry(
        retry=retry_if_exception(is_retryable_request),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    async def create_collection(
        self,
        collection_name: str,
        payload_schema=None,
    ):
        """
        Create a new collection in the Weaviate database if it does not already exist.

        The collection will be initialized with a default schema.

        Parameters:
        -----------

            - collection_name (str): The name of the new collection to be created.
            - payload_schema: Optional schema definition for the collection payload. (default
              None)

        Returns:
        --------

            The created collection's configuration, if a new collection was made, otherwise
            information about the existing collection.
        """
        import weaviate.classes.config as wvcc

        if not await self.has_collection(collection_name):
            client = await self.get_client()
            return await client.collections.create(
                name=collection_name,
                properties=[
                    wvcc.Property(
                        name="text", data_type=wvcc.DataType.TEXT, skip_vectorization=True
                    )
                ],
            )
        else:
            return await self.get_collection(collection_name)

    async def get_collection(self, collection_name: str):
        """
        Retrieve a collection from the Weaviate database by its name.

        Raise a CollectionNotFoundError if the specified collection does not exist.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to be retrieved.

        Returns:
        --------

            The requested collection object from the database.
        """
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        client = await self.get_client()
        return client.collections.get(collection_name)

    @retry(
        retry=retry_if_exception(is_retryable_request),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    @override_distributed(queued_add_data_points)
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """
        Create or update data points in the specified collection in the Weaviate database.

        Process the list of data points, embedding them and either inserting them or updating if
        they already exist.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to add data points to.
            - data_points (List[DataPoint]): A list of DataPoint objects to be created or
              updated in the collection.

        Returns:
        --------

            Information about the inserted or updated data points in the collection.
        """
        from weaviate.classes.data import DataObject

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        def convert_to_weaviate_data_points(data_point: DataPoint):
            """
            Transform a DataPoint object into a Weaviate DataObject format for insertion.

            Return a DataObject ready for use in Weaviate with the properties and vector included.

            Parameters:
            -----------

                - data_point (DataPoint): The DataPoint to convert into the Weaviate DataObject
                  format.

            Returns:
            --------

                The corresponding Weaviate DataObject representing the data point.
            """
            vector = data_vectors[data_points.index(data_point)]
            properties = data_point.model_dump()

            if "id" in properties:
                properties["uuid"] = str(data_point.id)
                del properties["id"]

            return DataObject(uuid=data_point.id, properties=properties, vector=vector)

        data_points = [convert_to_weaviate_data_points(data_point) for data_point in data_points]

        collection = await self.get_collection(collection_name)

        try:
            if len(data_points) > 1:
                return await collection.data.insert_many(data_points)
                # with collection.batch.dynamic() as batch:
                #     for data_point in data_points:
                #         batch.add_object(
                #             uuid=data_point.uuid,
                #             vector=data_point.vector,
                #             properties=data_point.properties,
                #             references=data_point.references,
                #         )
            else:
                data_point: DataObject = data_points[0]
                if await collection.data.exists(data_point.uuid):
                    return await collection.data.update(
                        uuid=data_point.uuid,
                        vector=data_point.vector,
                        properties=data_point.properties,
                        references=data_point.references,
                    )
                else:
                    return await collection.data.insert(
                        uuid=data_point.uuid,
                        vector=data_point.vector,
                        properties=data_point.properties,
                        references=data_point.references,
                    )
        except Exception as error:
            logger.error("Error creating data points: %s", str(error))
            raise error

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """
        Create a vector index based on an index name and property name by creating a
        corresponding collection.

        Parameters:
        -----------

            - index_name (str): The name for the vector index.
            - index_property_name (str): The property name associated with the vector index.

        Returns:
        --------

            The created collection representing the vector index.
        """
        return await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """
        Index a list of data points by creating an associated vector index collection.

        Data points are transformed into embeddable data before being processed for indexing.

        Parameters:
        -----------

            - index_name (str): The index name under which to store the data points.
            - index_property_name (str): The associated property name for the index.
            - data_points (list[DataPoint]): A list of DataPoint objects to be indexed.

        Returns:
        --------

            Information about the operation of indexing the data points.
        """
        return await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
                )
                for data_point in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """
        Fetch data points from a specified collection based on their IDs.

        Return data points wrapped in an object containing their properties after
        transformation.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to retrieve data points from.
            - data_point_ids (list[str]): A list of IDs for the data points to retrieve.

        Returns:
        --------

            A list of objects representing the retrieved data points.
        """
        from weaviate.classes.query import Filter

        collection = await self.get_collection(collection_name)
        data_points = await collection.query.fetch_objects(
            filters=Filter.by_id().contains_any(data_point_ids)
        )

        for data_point in data_points.objects:
            data_point.payload = data_point.properties
            data_point.id = data_point.uuid
            del data_point.properties

        return data_points.objects

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        """
        Perform a search on a collection using either a text query or a vector query.

        Return scored results based on the search criteria provided. Raise InvalidValueError if
        no query is provided.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to search within.
            - query_text (Optional[str]): Optional plain text query for searching. (default
              None)
            - query_vector (Optional[List[float]]): Optional vector representation for
              searching. (default None)
            - limit (int): The maximum number of results to return. (default 15)
            - with_vector (bool): Include vector information in the results. (default False)

        Returns:
        --------

            A list of scored results matching the search criteria.
        """
        import weaviate.classes as wvc
        import weaviate.exceptions

        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]

        collection = await self.get_collection(collection_name)

        try:
            search_result = await collection.query.hybrid(
                query=None,
                vector=query_vector,
                limit=limit if limit > 0 else None,
                include_vector=with_vector,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )

            return [
                ScoredResult(
                    id=parse_id(str(result.uuid)),
                    payload=result.properties,
                    score=1 - float(result.metadata.score),
                )
                for result in search_result.objects
            ]
        except weaviate.exceptions.WeaviateInvalidInputError:
            # Ignore if the collection doesn't exist
            return []

    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        """
        Execute a batch search for multiple query texts in the specified collection.

        Return a list of results for each query performed in parallel.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to search within.
            - query_texts (List[str]): A list of text queries to be processed in a batch.
            - limit (int): The maximum number of results to return for each query.
            - with_vectors (bool): Indicate whether to include vector information in the
              results. (default False)

        Returns:
        --------

            A list containing results for each search query executed.
        """

        def query_search(query_vector):
            """
            Wrap the search operation based on a query vector for fetching results.

            This function coordinates the search call, ensuring the collection name and search
            parameters are applied.

            Parameters:
            -----------

                - query_vector: The vector representation of the query for searching.

            Returns:
            --------

                The results of the search operation on the specified collection.
            """
            return self.search(
                collection_name, query_vector=query_vector, limit=limit, with_vector=with_vectors
            )

        return [
            await query_search(query_vector) for query_vector in await self.embed_data(query_texts)
        ]

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        """
        Remove specified data points from a collection based on their IDs.

        Return information about the deletion result, ideally confirming the operation's
        success.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete data
              points.
            - data_point_ids (list[str]): A list of IDs for the data points to be deleted.

        Returns:
        --------

            Confirmation of deletion operation result.
        """
        from weaviate.classes.query import Filter

        collection = await self.get_collection(collection_name)
        result = await collection.data.delete_many(
            filters=Filter.by_id().contains_any(data_point_ids)
        )

        return result

    async def prune(self):
        """
        Delete all collections from the Weaviate database.

        This operation will remove all data and cannot be undone.
        """
        client = await self.get_client()
        await client.collections.delete_all()
