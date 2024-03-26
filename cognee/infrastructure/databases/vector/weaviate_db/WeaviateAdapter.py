from typing import List, Optional
from multiprocessing import Pool
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wvcc
from weaviate.classes.data import DataObject
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..models.ScoredResult import ScoredResult
from ..embeddings.EmbeddingEngine import EmbeddingEngine


class WeaviateAdapter(VectorDBInterface):
    async_pool: Pool = None
    embedding_engine: EmbeddingEngine = None

    def __init__(self, url: str, api_key: str, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine

        self.client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
            # headers = {
            #     "X-OpenAI-Api-Key": openai_api_key
            # },
            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=30))
        )

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def create_collection(self, collection_name: str):
        return self.client.collections.create(
            name=collection_name,
            properties=[
                wvcc.Property(
                    name="text",
                    data_type=wvcc.DataType.TEXT,
                    skip_vectorization=True
                )
            ]
        )

    def get_collection(self, collection_name: str):
        return self.client.collections.get(collection_name)

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        data_vectors = await self.embed_data(
            list(map(lambda data_point: data_point.get_embeddable_data(), data_points)))

        def convert_to_weaviate_data_points(data_point: DataPoint):
            return DataObject(
                uuid=data_point.id,
                properties=data_point.payload,
                vector=data_vectors[data_points.index(data_point)]
            )

        objects = list(map(convert_to_weaviate_data_points, data_points))

        return self.get_collection(collection_name).data.insert_many(objects)

    async def search(
            self,
            collection_name: str,
            query_text: Optional[str] = None,
            query_vector: Optional[List[float]] = None,
            limit: int = None,
            with_vector: bool = False
    ):
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        search_result = self.get_collection(collection_name).query.hybrid(
            query=None,
            vector=query_vector if query_vector is not None else (await self.embed_data([query_text]))[0],
            limit=limit,
            include_vector=with_vector,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        return list(map(lambda result: ScoredResult(
            id=str(result.uuid),
            payload=result.properties,
            score=float(result.metadata.score)
        ), search_result.objects))

    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False):
        def query_search(query_vector):
            return self.search(collection_name, query_vector=query_vector, limit=limit, with_vector=with_vectors)

        return [await query_search(query_vector) for query_vector in await self.embed_data(query_texts)]

    async def prune(self):
        self.client.collections.delete_all()