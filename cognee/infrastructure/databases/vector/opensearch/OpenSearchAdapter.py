from typing import List, Optional
from opensearchpy import AsyncOpenSearch, NotFoundError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
import asyncio
import base64
import json

class IndexSchema(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}

class OpenSearchAdapter(VectorDBInterface):
    def __init__(
        self,
        hosts: list,
        embedding_engine: EmbeddingEngine,
        parameters_base64_key: Optional[str] = None,
    ):
        http_auth = None
        index_prefix=""
        use_ssl = False
        verify_certs = True
        ssl_assert_hostname = True
        ssl_show_warn = True

        #Decoding the parameters_key if provided
        if parameters_base64_key:
            vector_db_key_decoded = base64.b64decode(parameters_base64_key).decode("utf-8")
            vector_db_key_decoded_dict = json.loads(vector_db_key_decoded)
            username = vector_db_key_decoded_dict.get("username", None)
            password = vector_db_key_decoded_dict.get("password", None)
            if username and password:
                http_auth = (username, password)
            use_ssl = vector_db_key_decoded_dict.get("use_ssl", "False").lower() == "true"
            verify_certs = vector_db_key_decoded_dict.get("verify_certs", "True").lower() == "true"
            ssl_assert_hostname = vector_db_key_decoded_dict.get("ssl_assert_hostname", "True").lower() == "true"
            ssl_show_warn = vector_db_key_decoded_dict.get("ssl_show_warn", "True").lower() == "true"
            index_prefix = vector_db_key_decoded_dict.get("index_prefix", "")
        
        self.embedding_engine = embedding_engine
        self.index_prefix =f"{index_prefix}cognee"
        self.client = AsyncOpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            **{
                "use_ssl": use_ssl,
                "verify_certs": verify_certs,
                "ssl_assert_hostname": ssl_assert_hostname,
                "ssl_show_warn": ssl_show_warn,
            }
        )

    def _index_name(self, collection_name: str) -> str:
        return f"{self.index_prefix}_{collection_name}".lower()

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        index = self._index_name(collection_name)
        try:
            exists = await self.client.indices.exists(index=index)
            return exists
        except Exception:
            return False

    async def create_collection(self, collection_name: str, payload_schema=None):
        index = self._index_name(collection_name)
        if not await self.has_collection(collection_name):
            vector_size = self.embedding_engine.get_vector_size()
            body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "payload": {"type": "object"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": vector_size,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib"
                            }
                        }
                    }
                }
            }
            await self.client.indices.create(index=index, body=body)

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}")
    
    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
                )
                for data_point in data_points
            ],
        )

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        index = self._index_name(collection_name)
        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, type(data_points[0]))
        vectors = await self.embed_data([DataPoint.get_embeddable_data(dp) for dp in data_points])
        actions = []
        for i, dp in enumerate(data_points):
            doc = {
                "id": str(dp.id),
                "payload": dp.model_dump(),
                "vector": vectors[i]
            }
            actions.append({"index": {"_index": index, "_id": str(dp.id)}})
            actions.append(doc)
        # Bulk insert
        await self.client.bulk(body=actions, refresh=True)

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        index = self._index_name(collection_name)
        docs = []
        for id_ in data_point_ids:
            try:
                res = await self.client.get(index=index, id=id_)
                source = res["_source"]
                docs.append(
                    ScoredResult(
                        id=parse_id(source["id"]),
                        payload=source["payload"],
                        score=0
                    )
                )
            except NotFoundError:
                continue
        return docs

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")
        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]
        index = self._index_name(collection_name)
        query = {
            "size": limit,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": limit
                    }
                }
            }
        }
        try:
            res = await self.client.search(index=index, body=query)
            hits = res["hits"]["hits"]
            results = []
            for hit in hits:
                source = hit["_source"]
                score = 1 - hit.get("_score", 0)
                results.append(
                    ScoredResult(
                        id=parse_id(source["id"]),
                        payload=source["payload"],
                        score=score
                    )
                )
            return results
        except NotFoundError:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = 15,
        with_vectors: bool = False,
    ):
        vectors = await self.embed_data(query_texts)
        tasks = [
            self.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                with_vector=with_vectors
            )
            for vector in vectors
        ]
        return await asyncio.gather(*tasks)

    async def delete_data_points(self, collection_name: str, data_point_ids: List[str]):
        index = self._index_name(collection_name)
        actions = []
        for id_ in data_point_ids:
            actions.append({"delete": {"_index": index, "_id": id_}})
        await self.client.bulk(body=actions, refresh=True)

    async def prune(self):
        # Remove all indices with the prefix
        indices = await self.client.indices.get(index=f"{self.index_prefix}_*")
        for index in indices:
            await self.client.indices.delete(index=index)
