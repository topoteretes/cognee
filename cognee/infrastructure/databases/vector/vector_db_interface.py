from typing import List, Protocol, Optional
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
        payload_schema: Optional[PayloadSchema] = None,
    ):
        """
        Create a new collection with an optional payload schema.

        Parameters:
        -----------

            - collection_name (str): The name of the new collection to create.
            - payload_schema (Optional[PayloadSchema]): An optional schema for the payloads
              within this collection. (default None)
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
            - data_point_ids (list[str]): A list of IDs of the data points to retrieve.
        """
        raise NotImplementedError

    """ Search """

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        limit: int,
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
            - limit (int): The maximum number of results to return from the search.
            - with_vector (bool): Whether to return the vector representations with search
              results. (default False)
        """
        raise NotImplementedError

    @abstractmethod
    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        """
        Perform a batch search using multiple text queries against a collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to conduct the batch search in.
            - query_texts (List[str]): A list of text queries to use for the search.
            - limit (int): The maximum number of results to return for each query.
            - with_vectors (bool): Whether to include vector representations with search
              results. (default False)
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        """
        Delete specified data points from a collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete data
              points.
            - data_point_ids (list[str]): A list of IDs of the data points to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def prune(self):
        """
        Remove obsolete or unnecessary data from the database.
        """
        raise NotImplementedError
