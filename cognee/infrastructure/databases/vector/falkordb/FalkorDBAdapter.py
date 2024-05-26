
from typing import List, Dict, Optional, Any

from falkordb import FalkorDB
from qdrant_client import AsyncQdrantClient, models
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..embeddings.EmbeddingEngine import EmbeddingEngine




class FalcorDBAdapter(VectorDBInterface):
    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: str,
        graph_database_password: str,
        graph_database_port: int,
        driver: Optional[Any] = None,
        embedding_engine =  EmbeddingEngine,
        graph_name: str = "DefaultGraph",
    ):
        self.driver = FalkorDB(
            host = graph_database_url,
            port = graph_database_port)
        self.graph_name = graph_name
        self.embedding_engine = embedding_engine



    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)


    async def create_collection(self, collection_name: str, payload_schema = None):
        pass


    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        pass

    async def retrieve(self, collection_name: str, data_point_id: str):
        pass

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: int = 10,
        with_vector: bool = False,
    ):
        pass