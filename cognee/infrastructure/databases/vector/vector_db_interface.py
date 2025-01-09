from typing import List, Protocol, Optional
from abc import abstractmethod
from cognee.infrastructure.engine import DataPoint
from .models.PayloadSchema import PayloadSchema


class VectorDBInterface(Protocol):
    """Collections"""

    @abstractmethod
    async def has_collection(self, collection_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[PayloadSchema] = None,
    ):
        raise NotImplementedError

    """ Data points """

    @abstractmethod
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        raise NotImplementedError

    @abstractmethod
    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
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
        raise NotImplementedError

    @abstractmethod
    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        raise NotImplementedError

    @abstractmethod
    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        raise NotImplementedError

    @abstractmethod
    async def prune(self):
        raise NotImplementedError
