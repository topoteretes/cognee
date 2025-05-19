from __future__ import annotations
import asyncio
from uuid import UUID
from typing import List, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..vector_db_interface import VectorDBInterface

logger = get_logger("MilvusAdapter")


class IndexSchema(DataPoint):
    text: str

    metadata: dict = {"index_fields": ["text"]}


class MilvusAdapter(VectorDBInterface):
    name = "Milvus"
    url: str
    api_key: Optional[str]
    embedding_engine: EmbeddingEngine = None

    def __init__(self, url: str, api_key: Optional[str], embedding_engine: EmbeddingEngine):
        self.url = url
        self.api_key = api_key

        self.embedding_engine = embedding_engine

    def get_milvus_client(self):
        from pymilvus import MilvusClient

        if self.api_key:
            client = MilvusClient(uri=self.url, token=self.api_key)
        else:
            client = MilvusClient(uri=self.url)
        return client

    async def embed_data(self, data: List[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        future = asyncio.Future()
        client = self.get_milvus_client()
        future.set_result(client.has_collection(collection_name=collection_name))

        return await future

    async def create_collection(
        self,
        collection_name: str,
        payload_schema=None,
    ):
        from pymilvus import DataType, MilvusException

        client = self.get_milvus_client()
        if client.has_collection(collection_name=collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            return True

        try:
            dimension = self.embedding_engine.get_vector_size()
            assert dimension > 0, "Embedding dimension must be greater than 0."

            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )

            schema.add_field(
                field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=36
            )

            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=60535)

            index_params = client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="COSINE")

            client.create_collection(
                collection_name=collection_name, schema=schema, index_params=index_params
            )

            client.load_collection(collection_name)

            logger.info(f"Collection '{collection_name}' created successfully.")
            return True
        except MilvusException as e:
            logger.error(f"Error creating collection '{collection_name}': {str(e)}")
            raise e

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        from pymilvus import MilvusException, exceptions

        client = self.get_milvus_client()
        data_vectors = await self.embed_data(
            [data_point.get_embeddable_data(data_point) for data_point in data_points]
        )

        insert_data = [
            {
                "id": str(data_point.id),
                "vector": data_vectors[index],
                "text": data_point.text,
            }
            for index, data_point in enumerate(data_points)
        ]

        try:
            result = client.insert(collection_name=collection_name, data=insert_data)
            logger.info(
                f"Inserted {result.get('insert_count', 0)} data points into collection '{collection_name}'."
            )
            return result
        except exceptions.CollectionNotExistException as error:
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' does not exist!"
            ) from error
        except MilvusException as e:
            logger.error(
                f"Error inserting data points into collection '{collection_name}': {str(e)}"
            )
            raise e

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        formatted_data_points = [
            IndexSchema(
                id=data_point.id,
                text=getattr(data_point, data_point.metadata["index_fields"][0]),
            )
            for data_point in data_points
        ]
        collection_name = f"{index_name}_{index_property_name}"
        await self.create_data_points(collection_name, formatted_data_points)

    async def retrieve(self, collection_name: str, data_point_ids: list[UUID]):
        from pymilvus import MilvusException, exceptions

        client = self.get_milvus_client()
        try:
            filter_expression = f"""id in [{", ".join(f'"{id}"' for id in data_point_ids)}]"""

            results = client.query(
                collection_name=collection_name,
                expr=filter_expression,
                output_fields=["*"],
            )
            return results
        except exceptions.CollectionNotExistException as error:
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' does not exist!"
            ) from error
        except MilvusException as e:
            logger.error(
                f"Error retrieving data points from collection '{collection_name}': {str(e)}"
            )
            raise e

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        from pymilvus import MilvusException, exceptions

        client = self.get_milvus_client()
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        try:
            query_vector = query_vector or (await self.embed_data([query_text]))[0]

            output_fields = ["id", "text"]
            if with_vector:
                output_fields.append("vector")

            results = client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=limit if limit > 0 else None,
                output_fields=output_fields,
                search_params={
                    "metric_type": "COSINE",
                },
            )

            return [
                ScoredResult(
                    id=parse_id(result["id"]),
                    score=result["distance"],
                    payload=result.get("entity", {}),
                )
                for result in results[0]
            ]
        except exceptions.CollectionNotExistException as error:
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' does not exist!"
            ) from error
        except MilvusException as e:
            logger.error(f"Error during search in collection '{collection_name}': {str(e)}")
            raise e

    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        query_vectors = await self.embed_data(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        from pymilvus import MilvusException

        client = self.get_milvus_client()
        try:
            filter_expression = f"""id in [{", ".join(f'"{id}"' for id in data_point_ids)}]"""

            delete_result = client.delete(collection_name=collection_name, filter=filter_expression)

            logger.info(
                f"Deleted data points with IDs {data_point_ids} from collection '{collection_name}'."
            )
            return delete_result
        except MilvusException as e:
            logger.error(
                f"Error deleting data points from collection '{collection_name}': {str(e)}"
            )
            raise e

    async def prune(self):
        client = self.get_milvus_client()
        if client:
            collections = client.list_collections()
            for collection_name in collections:
                client.drop_collection(collection_name=collection_name)
            client.close()
