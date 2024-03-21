from typing import List
from multiprocessing import Pool
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wvcc
from weaviate.classes.data import DataObject
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..models.ScoredResult import ScoredResult

class WeaviateAdapter(VectorDBInterface):
    async_pool: Pool = None

    def __init__(self, url: str, api_key: str, openai_api_key: str):
        self.client = weaviate.connect_to_wcs(
            cluster_url = url,
            auth_credentials = weaviate.auth.AuthApiKey(api_key),
            headers = {
                "X-OpenAI-Api-Key": openai_api_key
            },
            additional_config = wvc.init.AdditionalConfig(timeou = wvc.init.Timeout(init = 30))
        )

    async def create_collection(self, collection_name: str, collection_config: dict):
        return self.client.collections.create(
            name = collection_name,
            vectorizer_config = wvcc.Configure.Vectorizer.text2vec_openai(),
            generative_config = wvcc.Configure.Generative.openai(),
            properties = [
                wvcc.Property(
                    name = "text",
                    data_type = wvcc.DataType.TEXT
                )
            ]
        )

    def get_collection(self, collection_name: str):
        return self.client.collections.get(collection_name)

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        def convert_to_weaviate_data_points(data_point: DataPoint):
            return DataObject(
                uuid = data_point.id,
                properties = data_point.payload
            )

        objects = list(map(convert_to_weaviate_data_points, data_points))

        return self.get_collection(collection_name).data.insert_many(objects)

    async def search(self, collection_name: str, query_text: str, limit: int, with_vector: bool = False):
        search_result = self.get_collection(collection_name).query.bm25(
            query = query_text,
            limit = limit,
            include_vector = with_vector,
            return_metadata = wvc.query.MetadataQuery(score = True),
        )

        return list(map(lambda result: ScoredResult(
            id = result.uuid,
            payload = result.properties,
            score = str(result.metadata.score)
        ), search_result.objects))

    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int,  with_vectors: bool = False):
        def query_search(query_text):
            return self.search(collection_name, query_text, limit = limit, with_vector = with_vectors)

        return [await query_search(query_text) for query_text in query_texts]

    async def prune(self):
        self.client.collections.delete_all()
