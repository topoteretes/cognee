from typing import List
from abc import abstractmethod
from typing import Protocol

class GraphDBInterface(Protocol):
    """ Graphs """
    @abstractmethod
    async def create_graph(
        self,
        graph_name: str,
        graph_config: object
    ): raise NotImplementedError

    @abstractmethod
    async def update_graph(
        self,
        collection_name: str,
        collection_config: object
    ): raise NotImplementedError

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
        data_points: List[any]
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
