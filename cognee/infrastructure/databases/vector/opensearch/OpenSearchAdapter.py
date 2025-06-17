from typing import List, Optional
from opensearchpy import AsyncOpenSearch, NotFoundError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
import asyncio

class IndexSchema(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}

class OpenSearchAdapter(VectorDBInterface):
    def __init__(
        self,
        hosts: list,
        embedding_engine: EmbeddingEngine,
        http_auth: Optional[tuple] = None,
        index_prefix: str = "cognee",
        **kwargs
    ):
        self.embedding_engine = embedding_engine
        self.index_prefix = index_prefix
        self.client = AsyncOpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            **kwargs
        )

    def _index_name(self, collection_name: str) -> str:
        return f"{self.index_prefix}_{collection_name}"

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
                score = hit.get("_score", 0)
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
