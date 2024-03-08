from typing import List
from abc import abstractmethod
from typing import Protocol

class GraphDBInterface(Protocol):

    """ Save and Load Graphs """

    @abstractmethod
    async def save_graph(
        self,
        path: str
    ): raise NotImplementedError

    @abstractmethod
    async def load_graph(
        self,
        path: str
    ): raise NotImplementedError

    @abstractmethod
    async def delete_graph(
        self,
        path: str
    ): raise NotImplementedError

    """ CRUD operations on graph nodes """

    @abstractmethod
    async def create(self,
                     user_id:str,
                     custom_user_properties:str,
                     required_layers:list,
                     default_fields:dict
    ): raise NotImplementedError

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
