from typing import List, Protocol, Optional, Union, Any
from abc import abstractmethod
from cognee.infrastructure.engine import DataPoint
from .models.PayloadSchema import PayloadSchema


class VectorDBInterface(Protocol):
    """
    Defines an interface for interacting with a vector database, including operations for
    managing collections and data points.
    """

    @abstractmethod
    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: True if the collection exists, otherwise False.
        """
        raise NotImplementedError

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[Any] = None,
    ):
        """
        Create a new collection with an optional payload schema.

        Parameters:
        -----------

            - collection_name (str): The name of the new collection to create.
            - payload_schema (Optional[Any]): An optional schema for the payloads
              within this collection. Can be PayloadSchema, BaseModel, or other schema types. (default None)
        """
        raise NotImplementedError

    """ Data points """

    @abstractmethod
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """
        Insert new data points into the specified collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection where data points will be added.
            - data_points (List[DataPoint]): A list of data points to be added to the
              collection.
        """
        raise NotImplementedError

    @abstractmethod
    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """
        Retrieve data points from a collection using their IDs.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to retrieve data
              points.
            - data_point_ids (Union[List[str], list[str]]): A list of IDs of the data points to retrieve.
        """
        raise NotImplementedError

    """ Search """

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        limit: Optional[int],
        with_vector: bool = False,
    ):
        """
        Perform a search in the specified collection using either a text query or a vector
        query.

        Parameters:
        -----------

            - collection_name (str): The name of the collection in which to perform the search.
            - query_text (Optional[str]): An optional text query to search for in the
              collection.
            - query_vector (Optional[List[float]]): An optional vector representation for
              searching the collection.
            - limit (Optional[int]): The maximum number of results to return from the search.
            - with_vector (bool): Whether to return the vector representations with search
              results. (default False)
        """
        raise NotImplementedError

    @abstractmethod
    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int],
        with_vectors: bool = False,
    ):
        """
        Perform a batch search using multiple text queries against a collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to conduct the batch search in.
            - query_texts (List[str]): A list of text queries to use for the search.
            - limit (Optional[int]): The maximum number of results to return for each query.
            - with_vectors (bool): Whether to include vector representations with search
              results. (default False)
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_data_points(
        self, collection_name: str, data_point_ids: Union[List[str], list[str]]
    ):
        """
        Delete specified data points from a collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete data
              points.
            - data_point_ids (Union[List[str], list[str]]): A list of IDs of the data points to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def prune(self):
        """
        Remove obsolete or unnecessary data from the database.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_data(self, data: List[str]) -> List[List[float]]:
        """
        Embed textual data into vector representations.

        Parameters:
        -----------

            - data (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - List[List[float]]: A list of embedded vectors corresponding to the input data.
        """
        raise NotImplementedError

    # Optional methods that may be implemented by adapters
    async def get_connection(self):
        """
        Get a connection to the vector database.
        This method is optional and may return None for adapters that don't use connections.
        """
        return None

    async def get_collection(self, collection_name: str):
        """
        Get a collection object from the vector database.
        This method is optional and may return None for adapters that don't expose collection objects.
        """
        return None

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """
        Create a vector index for improved search performance.
        This method is optional and may be a no-op for adapters that don't support indexing.
        """
        pass

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        """
        Index data points for improved search performance.
        This method is optional and may be a no-op for adapters that don't support separate indexing.

        Parameters:
        -----------
            - index_name (str): Name of the index to create/update
            - index_property_name (str): Property name to index on
            - data_points (List[DataPoint]): Data points to index
        """
        pass

    def get_data_point_schema(self, model_type: Any) -> Any:
        """
        Get or transform a data point schema for the specific vector database.
        This method is optional and may return the input unchanged for simple adapters.

        Parameters:
        -----------
            - model_type (Any): The model type to get schema for

        Returns:
        --------
            - Any: The schema object suitable for this vector database
        """
        return model_type
