import json
from uuid import UUID
from typing import List, Optional
from chromadb import AsyncHttpClient, Settings

from cognee.exceptions import InvalidValueError
from cognee.shared.logging_utils import get_logger
from cognee.modules.storage.utils import get_own_properties
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..vector_db_interface import VectorDBInterface
from ..utils import normalize_distances

logger = get_logger("ChromaDBAdapter")


class IndexSchema(DataPoint):
    text: str

    metadata: dict = {"index_fields": ["text"]}

    def model_dump(self):
        data = super().model_dump()
        return process_data_for_chroma(data)


def process_data_for_chroma(data):
    """Convert complex data types to a format suitable for ChromaDB storage."""
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, UUID):
            processed_data[key] = str(value)
        elif isinstance(value, dict):
            # Store dictionaries as JSON strings with special prefix
            processed_data[f"{key}__dict"] = json.dumps(value)
        elif isinstance(value, list):
            # Store lists as JSON strings with special prefix
            processed_data[f"{key}__list"] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            processed_data[key] = value
        else:
            processed_data[key] = str(value)
    return processed_data


def restore_data_from_chroma(data):
    """Restore original data structure from ChromaDB storage format."""
    restored_data = {}
    dict_keys = []
    list_keys = []

    # First, identify all special keys
    for key in data.keys():
        if key.endswith("__dict"):
            dict_keys.append(key)
        elif key.endswith("__list"):
            list_keys.append(key)
        else:
            restored_data[key] = data[key]

    # Process dictionary fields
    for key in dict_keys:
        original_key = key[:-6]  # Remove '__dict' suffix
        try:
            restored_data[original_key] = json.loads(data[key])
        except Exception as e:
            logger.debug(f"Error restoring dictionary from JSON: {e}")
            restored_data[key] = data[key]

    # Process list fields
    for key in list_keys:
        original_key = key[:-6]  # Remove '__list' suffix
        try:
            restored_data[original_key] = json.loads(data[key])
        except Exception as e:
            logger.debug(f"Error restoring list from JSON: {e}")
            restored_data[key] = data[key]

    return restored_data


class ChromaDBAdapter(VectorDBInterface):
    name = "ChromaDB"
    url: str
    api_key: str
    connection: AsyncHttpClient = None

    def __init__(
        self, url: Optional[str], api_key: Optional[str], embedding_engine: EmbeddingEngine
    ):
        self.embedding_engine = embedding_engine
        self.url = url
        self.api_key = api_key

    async def get_connection(self) -> AsyncHttpClient:
        if self.connection is None:
            settings = Settings(
                chroma_client_auth_provider="token", chroma_client_auth_credentials=self.api_key
            )
            self.connection = await AsyncHttpClient(host=self.url, settings=settings)

        return self.connection

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        collections = await self.get_collection_names()
        return collection_name in collections

    async def create_collection(self, collection_name: str, payload_schema=None):
        client = await self.get_connection()

        if not await self.has_collection(collection_name):
            await client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    async def get_collection(self, collection_name: str) -> AsyncHttpClient:
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")

        client = await self.get_connection()
        return await client.get_collection(collection_name)

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        await self.create_collection(collection_name)

        collection = await self.get_collection(collection_name)

        texts = [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        embeddings = await self.embed_data(texts)
        ids = [str(data_point.id) for data_point in data_points]

        metadatas = []
        for data_point in data_points:
            metadata = get_own_properties(data_point)
            metadatas.append(process_data_for_chroma(metadata))

        await collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
        )

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create a vector index as a ChromaDB collection."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """Index data points using the specified index property."""
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """Retrieve data points by their IDs from a collection."""
        collection = await self.get_collection(collection_name)
        results = await collection.get(ids=data_point_ids, include=["metadatas"])

        return [
            ScoredResult(
                id=parse_id(id),
                payload=restore_data_from_chroma(metadata),
                score=0,
            )
            for id, metadata in zip(results["ids"], results["metadatas"])
        ]

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: int = 15,
        with_vector: bool = False,
        normalized: bool = True,
    ):
        """Search for similar items in a collection using text or vector query."""
        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        try:
            collection = await self.get_collection(collection_name)

            if limit == 0:
                limit = await collection.count()

            results = await collection.query(
                query_embeddings=[query_vector],
                include=["metadatas", "distances", "embeddings"]
                if with_vector
                else ["metadatas", "distances"],
                n_results=limit,
            )

            vector_list = []
            for i, (id, metadata, distance) in enumerate(
                zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
            ):
                item = {
                    "id": parse_id(id),
                    "payload": restore_data_from_chroma(metadata),
                    "_distance": distance,
                }

                if with_vector and "embeddings" in results:
                    item["vector"] = results["embeddings"][0][i]

                vector_list.append(item)

            # Normalize vector distance
            normalized_values = normalize_distances(vector_list)
            for i in range(len(normalized_values)):
                vector_list[i]["score"] = normalized_values[i]

            # Create and return ScoredResult objects
            return [
                ScoredResult(
                    id=row["id"],
                    payload=row["payload"],
                    score=row["score"],
                    vector=row.get("vector") if with_vector else None,
                )
                for row in vector_list
            ]
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = 5,
        with_vectors: bool = False,
    ):
        """Perform multiple searches in a single request for efficiency."""
        query_vectors = await self.embed_data(query_texts)

        collection = await self.get_collection(collection_name)

        results = await collection.query(
            query_embeddings=query_vectors,
            include=["metadatas", "distances", "embeddings"]
            if with_vectors
            else ["metadatas", "distances"],
            n_results=limit,
        )

        all_results = []
        for i in range(len(query_texts)):
            vector_list = []

            for j, (id, metadata, distance) in enumerate(
                zip(results["ids"][i], results["metadatas"][i], results["distances"][i])
            ):
                item = {
                    "id": parse_id(id),
                    "payload": restore_data_from_chroma(metadata),
                    "_distance": distance,
                }

                if with_vectors and "embeddings" in results:
                    item["vector"] = results["embeddings"][i][j]

                vector_list.append(item)

            normalized_values = normalize_distances(vector_list)

            query_results = []
            for j, item in enumerate(vector_list):
                result = ScoredResult(
                    id=item["id"],
                    payload=item["payload"],
                    score=normalized_values[j],
                )

                if with_vectors and "embeddings" in results:
                    result.vector = item.get("vector")

                query_results.append(result)

            all_results.append(query_results)

        return all_results

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        """Remove data points from a collection by their IDs."""
        collection = await self.get_collection(collection_name)
        await collection.delete(ids=data_point_ids)
        return True

    async def prune(self):
        """Delete all collections in the ChromaDB database."""
        client = await self.get_connection()
        collections = await self.list_collections()
        for collection_name in collections:
            await client.delete_collection(collection_name)
        return True

    async def get_collection_names(self):
        """Get a list of all collection names in the database."""
        client = await self.get_connection()
        collections = await client.list_collections()
        return [
            collection.name if hasattr(collection, "name") else collection["name"]
            for collection in collections
        ]
