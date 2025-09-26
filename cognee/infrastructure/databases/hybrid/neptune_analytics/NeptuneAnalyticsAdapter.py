"""Neptune Analytics Hybrid Adapter combining Vector and Graph functionality"""

import asyncio
import json
from typing import List, Optional, Any, Dict, Type, Tuple
from uuid import UUID

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.exceptions import MutuallyExclusiveQueryParametersError
from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.models.PayloadSchema import PayloadSchema
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult

logger = get_logger("NeptuneAnalyticsAdapter")


class IndexSchema(DataPoint):
    """
    Represents a schema for an index data point containing an ID and text.

    Attributes:
    - id: A string representing the unique identifier for the data point.
    - text: A string representing the content of the data point.
    - metadata: A dictionary with default index fields for the schema, currently configured
    to include 'text'.
    """

    id: str
    text: str
    metadata: dict = {"index_fields": ["text"]}


NEPTUNE_ANALYTICS_ENDPOINT_URL = "neptune-graph://"


class NeptuneAnalyticsAdapter(NeptuneGraphDB, VectorDBInterface):
    """
    Hybrid adapter that combines Neptune Analytics Vector and Graph functionality.

    This adapter extends NeptuneGraphDB and implements VectorDBInterface to provide
    a unified interface for working with Neptune Analytics as both a vector store
    and a graph database.
    """

    _VECTOR_NODE_LABEL = "COGNEE_NODE"
    _COLLECTION_PREFIX = "VECTOR_COLLECTION"
    _TOPK_LOWER_BOUND = 0
    _TOPK_UPPER_BOUND = 10

    def __init__(
        self,
        graph_id: str,
        embedding_engine: Optional[EmbeddingEngine] = None,
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize the Neptune Analytics hybrid adapter.

        Parameters:
        -----------
            - graph_id (str): The Neptune Analytics graph identifier
            - embedding_engine(Optional[EmbeddingEngine]): The embedding engine instance to translate text to vector.
            - region (Optional[str]): AWS region where the graph is located (default: us-east-1)
            - aws_access_key_id (Optional[str]): AWS access key ID
            - aws_secret_access_key (Optional[str]): AWS secret access key
            - aws_session_token (Optional[str]): AWS session token for temporary credentials
        """
        # Initialize the graph database functionality
        super().__init__(
            graph_id=graph_id,
            region=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

        # Add vector-specific attributes
        self.embedding_engine = embedding_engine
        logger.info(
            f'Initialized Neptune Analytics hybrid adapter for graph: "{graph_id}" in region: "{self.region}"'
        )

    # VectorDBInterface methods implementation

    async def get_connection(self):
        """
        This method is part of the default implementation but not defined in the interface.
        No operation is performed and None will be returned here,
        because the concept of connection is not applicable in this context.
        """
        return None

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embeds the provided textual data into vector representation.

        Uses the embedding engine to convert the list of strings into a list of float vectors.

        Parameters:
        -----------
            - data (list[str]): A list of strings representing the data to be embedded.

        Returns:
        --------
            - list[list[float]]: A list of embedded vectors corresponding to the input data.
        """
        self._validate_embedding_engine()
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Neptune Analytics stores vector on a node level,
        so create_collection() implements interface for compliance but performs no operations when called.

        Parameters:
        -----------
            - collection_name (str): The name of the collection to check for existence.
        Returns:
        --------
            - bool: Always return True.
        """
        return True

    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[PayloadSchema] = None,
    ):
        """
        Neptune Analytics stores vector on a node level, so create_collection() implements interface for compliance but performs no operations when called.
        As the result, create_collection() will be no-op.

        Parameters:
        -----------
            - collection_name (str): The name of the new collection to create.
            - payload_schema (Optional[PayloadSchema]): An optional schema for the payloads
              within this collection. (default None)
        """
        pass

    async def get_collection(self, collection_name: str):
        """
        This method is part of the default implementation but not defined in the interface.
        No operation is performed here because the concept of collection is not applicable in NeptuneAnalytics vector store.
        """
        return None

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """
        Insert new data points into the specified collection, by first inserting the node itself on the graph,
        then execute neptune.algo.vectors.upsert() to insert the corresponded embedding.

        Parameters:
        -----------
            - collection_name (str): The name of the collection where data points will be added.
            - data_points (List[DataPoint]): A list of data points to be added to the
              collection.
        """
        self._validate_embedding_engine()

        # Fetch embeddings
        texts = [DataPoint.get_embeddable_data(t) for t in data_points]
        data_vectors = await self.embedding_engine.embed_text(texts)

        for index, data_point in enumerate(data_points):
            node_id = data_point.id
            # Fetch embedding from list instead
            data_vector = data_vectors[index]

            # Fetch properties
            properties = self._serialize_properties(data_point.model_dump())
            properties[self._COLLECTION_PREFIX] = collection_name
            params = dict(
                node_id=str(node_id),
                properties=properties,
                embedding=data_vector,
                collection_name=collection_name,
            )

            # Compose the query and send
            query_string = (
                f"MERGE (n "
                f":{self._VECTOR_NODE_LABEL} "
                f" {{`~id`: $node_id}}) "
                f"ON CREATE SET n = $properties, n.updated_at = timestamp() "
                f"ON MATCH SET n += $properties, n.updated_at = timestamp() "
                f"WITH n, $embedding AS embedding "
                f"CALL neptune.algo.vectors.upsert(n, embedding) "
                f"YIELD success "
                f"RETURN success "
            )

            try:
                self._client.query(query_string, params)
            except Exception as e:
                self._na_exception_handler(e, query_string)
        pass

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """
        Retrieve data points from a collection using their IDs.

        Parameters:
        -----------
            - collection_name (str): The name of the collection from which to retrieve data
              points.
            - data_point_ids (list[str]): A list of IDs of the data points to retrieve.
        """
        # Do the fetch for each node
        params = dict(node_ids=data_point_ids, collection_name=collection_name)
        query_string = (
            f"MATCH( n :{self._VECTOR_NODE_LABEL}) "
            f"WHERE id(n) in $node_ids AND "
            f"n.{self._COLLECTION_PREFIX} = $collection_name "
            f"RETURN n as payload "
        )

        try:
            result = self._client.query(query_string, params)
            return [self._get_scored_result(item) for item in result]
        except Exception as e:
            self._na_exception_handler(e, query_string)

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = None,
        with_vector: bool = False,
    ):
        """
        Perform a search in the specified collection using either a text query or a vector
        query.

        Parameters:
        -----------
            - collection_name (str): The name of the collection in which to perform the search.
            - query_text (Optional[str]): An optional text query to search for in the
              collection.
            - query_vector (Optional[List[float]]): An optional vector representation for
              searching the collection.
            - limit (int): The maximum number of results to return from the search.
            - with_vector (bool): Whether to return the vector representations with search
              results, this is not supported for Neptune Analytics backend at the moment.

        Returns:
        --------
            A list of scored results that match the query.
        """
        self._validate_embedding_engine()

        if with_vector:
            logger.warning(
                "with_vector=True will include embedding vectors in the result. "
                "This may trigger a resource-intensive query and increase response time. "
                "Use this option only when vector data is required."
            )

        # In the case of excessive limit, or None / zero / negative value, limit will be set to 10.
        if not limit or limit <= self._TOPK_LOWER_BOUND or limit > self._TOPK_UPPER_BOUND:
            logger.warning(
                "Provided limit (%s) is invalid (None, zero, negative, or exceeds maximum). "
                "Defaulting to limit=10.",
                limit,
            )
            limit = self._TOPK_UPPER_BOUND

        if query_vector and query_text:
            raise MutuallyExclusiveQueryParametersError()
        elif query_text is None and query_vector is None:
            raise MissingQueryParameterError()
        elif query_vector:
            embedding = query_vector
        else:
            data_vectors = await self.embedding_engine.embed_text([query_text])
            embedding = data_vectors[0]

        # Compose the parameters map
        params = dict(embedding=embedding, param_topk=limit)
        # Compose the query
        query_string = f"""
        CALL neptune.algo.vectors.topKByEmbeddingWithFiltering({{
                topK: {limit},
                embedding: {embedding}, 
                nodeFilter: {{ equals: {{property: '{self._COLLECTION_PREFIX}', value: '{collection_name}'}} }}
              }}
            )
        YIELD node, score
        """

        if with_vector:
            query_string += """
        WITH node, score, id(node) as node_id 
        MATCH (n)
        WHERE id(n) = id(node)
        CALL neptune.algo.vectors.get(n)
        YIELD embedding
        RETURN node as payload, score, embedding
        """

        else:
            query_string += """
        RETURN node as payload, score
        """

        try:
            query_response = self._client.query(query_string, params)
            return [self._get_scored_result(item=item, with_score=True) for item in query_response]
        except Exception as e:
            self._na_exception_handler(e, query_string)

    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        """
        Perform a batch search using multiple text queries against a collection.

        Parameters:
        -----------
            - collection_name (str): The name of the collection to conduct the batch search in.
            - query_texts (List[str]): A list of text queries to use for the search.
            - limit (int): The maximum number of results to return for each query.
            - with_vectors (bool): Whether to include vector representations with search
              results. (default False)

        Returns:
        --------
            A list of search result sets, one for each query input.
        """
        self._validate_embedding_engine()

        # Convert text to embedding array in batch
        data_vectors = await self.embedding_engine.embed_text(query_texts)
        return await asyncio.gather(
            *[
                self.search(collection_name, None, vector, limit, with_vectors)
                for vector in data_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        """
        Delete specified data points from a collection, by executing an OpenCypher query,
        with matching [vector_label, collection_label, node_id] combination.

        Parameters:
        -----------
            - collection_name (str): The name of the collection from which to delete data
              points.
            - data_point_ids (list[str]): A list of IDs of the data points to delete.
        """
        params = dict(node_ids=data_point_ids, collection_name=collection_name)
        query_string = (
            f"MATCH (n :{self._VECTOR_NODE_LABEL}) "
            f"WHERE id(n) IN $node_ids "
            f"AND n.{self._COLLECTION_PREFIX} = $collection_name "
            f"DETACH DELETE n"
        )
        try:
            self._client.query(query_string, params)
        except Exception as e:
            self._na_exception_handler(e, query_string)
        pass

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """
        Neptune Analytics stores vectors at the node level,
        so create_vector_index() implements the interface for compliance but performs no operation when called.
        As a result, create_vector_index() invokes create_collection(), which is also a no-op.
        This ensures the logic flow remains consistent, even if the concept of collections is introduced in a future release.
        """
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """
        Indexes a list of data points into Neptune Analytics by creating them as nodes.

        This method constructs a unique collection name by combining the `index_name` and
        `index_property_name`, then delegates to `create_data_points()` to store the data.

        Args:
            index_name (str): The base name of the index.
            index_property_name (str): The property name to append to the index name for uniqueness.
            data_points (list[DataPoint]): A list of `DataPoint` instances to be indexed.

        Returns:
            None
        """
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=str(data_point.id),
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def prune(self):
        """
        Remove obsolete or unnecessary data from the database.
        """
        # Run actual truncate
        self._client.query(f"MATCH (n :{self._VECTOR_NODE_LABEL}) DETACH DELETE n")
        pass

    @staticmethod
    def _get_scored_result(
        item: dict, with_vector: bool = False, with_score: bool = False
    ) -> ScoredResult:
        """
        Util method to simplify the object creation of ScoredResult base on incoming NX payload response.
        """
        return ScoredResult(
            id=item.get("payload").get("~id"),
            payload=item.get("payload").get("~properties"),
            score=item.get("score") if with_score else 0,
            vector=item.get("embedding") if with_vector else None,
        )

    def _na_exception_handler(self, ex, query_string: str):
        """
        Generic exception handler for NA langchain.
        """
        logger.error("Neptune Analytics query failed: %s | Query: [%s]", ex, query_string)
        raise ex

    def _validate_embedding_engine(self):
        """
        Validates if the embedding_engine is defined
        :raises: ValueError if this object does not have a valid embedding_engine
        """
        if self.embedding_engine is None:
            raise ValueError(
                "Neptune Analytics requires an embedder defined to make vector operations"
            )
