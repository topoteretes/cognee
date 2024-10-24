import asyncio
from falkordb import FalkorDB
from ..models.DataPoint import DataPoint
from ..vector_db_interface import VectorDBInterface
from ..embeddings.EmbeddingEngine import EmbeddingEngine


class FalcorDBAdapter(VectorDBInterface):
    def __init__(
        self,
        graph_database_url: str,
        graph_database_port: int,
        embedding_engine =  EmbeddingEngine,
    ):
        self.driver = FalkorDB(
            host = graph_database_url,
            port = graph_database_port)
        self.embedding_engine = embedding_engine


    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        collections = self.driver.list_graphs()

        return collection_name in collections

    async def create_collection(self, collection_name: str, payload_schema = None):
        self.driver.select_graph(collection_name)

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        graph = self.driver.select_graph(collection_name)

        def stringify_properties(properties: dict) -> str:
            return ",".join(f"{key}:'{value}'" for key, value in properties.items())
            
        def create_data_point_query(data_point: DataPoint):
            node_label = type(data_point.payload).__name__
            node_properties = stringify_properties(data_point.payload.dict())
          
            return f"""CREATE (:{node_label} {{{node_properties}}})"""

        query = " ".join([create_data_point_query(data_point) for data_point in data_points])

        graph.query(query)

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        graph = self.driver.select_graph(collection_name)

        return graph.query(
            f"MATCH (node) WHERE node.id IN $node_ids RETURN node",
            {
                "node_ids": data_point_ids,
            },
        )

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: list[float] = None,
        limit: int = 10,
        with_vector: bool = False,
    ):
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        graph = self.driver.select_graph(collection_name)

        query = f"""
            CALL db.idx.vector.queryNodes(
                null,
                'text',
                {limit},
                {query_vector}
            ) YIELD node, score
        """

        result = graph.query(query)

        return result

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[self.search(
                collection_name = collection_name,
                query_vector = query_vector,
                limit = limit,
                with_vector = with_vectors,
            ) for query_vector in query_vectors]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        graph = self.driver.select_graph(collection_name)

        return graph.query(
            f"MATCH (node) WHERE node.id IN $node_ids DETACH DELETE node",
            {
                "node_ids": data_point_ids,
            },
        )
