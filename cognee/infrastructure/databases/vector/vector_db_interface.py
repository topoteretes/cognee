from typing import List, Protocol, Optional
from abc import abstractmethod
from .models.DataPoint import DataPoint

class VectorDBInterface(Protocol):
    """ Collections """
    @abstractmethod
    async def create_collection(
        self,
        collection_name: str
    ): raise NotImplementedError

    # @abstractmethod
    # async def update_collection(
    #     self,
    #     collection_name: str,
    #     collection_config: object
    # ): raise NotImplementedError

    # @abstractmethod
    # async def delete_collection(
    #     self,
    #     collection_name: str
    # ): raise NotImplementedError

    # @abstractmethod
    # async def create_vector_index(
    #     self,
    #     collection_name: str,
    #     vector_index_config: object
    # ): raise NotImplementedError

    # @abstractmethod
    # async def create_data_index(
    #     self,
    #     collection_name: str,
    #     vector_index_config: object
    # ): raise NotImplementedError

    """ Data points """
    @abstractmethod
    async def create_data_points(
        self,
        collection_name: str,
        data_points: List[DataPoint]
    ): raise NotImplementedError

    # @abstractmethod
    # async def get_data_point(
    #     self,
    #     collection_name: str,
    #     data_point_id: str
    # ): raise NotImplementedError

    # @abstractmethod
    # async def update_data_point(
    #     self,
    #     collection_name: str,
    #     data_point_id: str,
    #     payload: object
    # ): raise NotImplementedError

    # @abstractmethod
    # async def delete_data_point(
    #     self,
    #     collection_name: str,
    #     data_point_id: str
    # ): raise NotImplementedError

    """ Search """
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        limit: int,
        with_vector: bool = False

    ): raise NotImplementedError

    @abstractmethod
    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int,
        with_vectors: bool = False
    ): raise NotImplementedError
