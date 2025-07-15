import asyncio
from typing import List, Optional
from langchain_aws import NeptuneAnalyticsGraph, NeptuneGraph

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import get_own_properties
from cognee.shared.logging_utils import get_logger
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.PayloadSchema import PayloadSchema
from ..models.ScoredResult import ScoredResult
from ..vector_db_interface import VectorDBInterface

logger = get_logger("NeptuneAnalyticsDBAdapter")

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


class NeptuneAnalyticsAdapter(VectorDBInterface):
    name = "Neptune Analytics"

    VECTOR_NODE_IDENTIFIER = "COGNEE_VECTOR_NODE"
    COLLECTION_PREFIX = "VECTOR_COLLECTION"

    def __init__(self,
                 graph_id: Optional[str],
                 embedding_engine: EmbeddingEngine,
                 region: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None,
                 ):
        """
        Initialize the Neptune Analytics vector database adapter.

        Parameters:
        -----------
            - graph_id (str): The Neptune Analytics graph identifier
            - embedding_engine(EmbeddingEngine): The embedding engine instance to translate text to vector.
            - region (Optional[str]): AWS region where the graph is located (default: us-east-1)
            - aws_access_key_id (Optional[str]): AWS access key ID
            - aws_secret_access_key (Optional[str]): AWS secret access key
            - aws_session_token (Optional[str]): AWS session token for temporary credentials

        """
        self.graph_id = graph_id
        self.embedding_engine = embedding_engine
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self._client = NeptuneAnalyticsGraph(graph_id)

    """ Collection related """

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
Neptune Analytics stores vector on a node level, so create_collection() implements interface for compliance but performs no operations when called.```
        as the result, create_collection( ) will be no-op.
        is available.

        Parameters:
        -----------
            - collection_name (str): The name of the new collection to create.
            - payload_schema (Optional[PayloadSchema]): An optional schema for the payloads
              within this collection. (default None)
        """
        pass

    """ Data points """

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
        # Fetch embeddings
        texts = [DataPoint.get_embeddable_data(t) for t in data_points]
        data_vectors = (await self.embedding_engine.embed_text(texts))

        for index, data_point in enumerate(data_points):
            node_id = data_point.id
            # Fetch embedding from list instead
            data_vector = data_vectors[index]

            # Fetch properties
            properties = get_own_properties(data_point)
            properties[self.COLLECTION_PREFIX] = collection_name
            params = dict(node_id = node_id, properties = properties,
                          embedding = data_vector, collection_name = collection_name)

            # Compose the query and send
            query_string = (
                    f"MERGE (n "
                    f":{self.VECTOR_NODE_IDENTIFIER} "
                    f" {{{self.COLLECTION_PREFIX}: $collection_name, `~id`: $node_id}}) "
                    f"SET n = $properties "
                    f"WITH n, $embedding AS embedding "
                    f"CALL neptune.algo.vectors.upsert(n, embedding) "
                    f"YIELD success "
                    f"RETURN success ")
            self._client.query(query_string, params)
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
        query_string = (f"MATCH( n :{self.VECTOR_NODE_IDENTIFIER}) "
                        f"WHERE id(n) in $node_ids AND "
                        f"n.{self.COLLECTION_PREFIX} = $collection_name "
                        f"RETURN n as payload ")
        result = self._client.query(query_string, params)

        result_set = [ScoredResult(
            id=item.get('payload').get('~id'),
            payload=item.get('payload').get('~properties'),
            score=0
        ) for item in result]
        return result_set

    """ Search """

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        limit: int,
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
        if with_vector:
            logger.warning(
                "with_vector=True will include embedding vectors in the result. "
                "This may trigger a resource-intensive query and increase response time. "
                "Use this option only when vector data is required."
            )

        if query_vector and query_text:
            raise InvalidValueError(
                message="The search function accepts either text or embedding as input, but not both."
            )
        elif query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")
        elif query_vector:
            embedding = query_vector
        else:
            data_vectors = (await self.embedding_engine.embed_text([query_text]))
            embedding = data_vectors[0]

        # Compose the parameters map
        params = dict(embedding=embedding, param_topk=limit)
        # Compose the query
        query_string = f"""
        CALL neptune.algo.vectors.topKByEmbeddingWithFiltering({{
                topK: {limit},
                embedding: {embedding}, 
                nodeFilter: {{ equals: {{property: '{self.COLLECTION_PREFIX}', value: '{collection_name}'}} }}
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

        query_response = self._client.query(query_string, params)
        return [ScoredResult(
            id=item.get('payload').get('~id'),
            payload=item.get('payload').get('~properties'),
            score=item.get('score'),
            vector=item.get('embedding') if with_vector else None
        ) for item in query_response]

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

        # Convert text to embedding array in batch
        data_vectors = (await self.embedding_engine.embed_text(query_texts))
        return await asyncio.gather(*[
            self.search(collection_name, None, vector, limit, with_vectors)
            for vector in data_vectors
        ])


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
        query_string = (f"MATCH (n :{self.VECTOR_NODE_IDENTIFIER}) "
                        f"WHERE id(n) IN $node_ids "
                        f"AND n.{self.COLLECTION_PREFIX} = $collection_name "
                        f"DETACH DELETE n")
        self._client.query(query_string, params)
        pass

    async def prune(self):
        """
        Remove obsolete or unnecessary data from the database.
        """
        # Run actual truncate
        self._client.query(f"MATCH (n :{self.VECTOR_NODE_IDENTIFIER}) "
                           f"DETACH DELETE n")
        pass

