from typing import List, Optional
from langchain_aws import NeptuneAnalyticsGraph, NeptuneGraph
from cognee.infrastructure.engine import DataPoint
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.PayloadSchema import PayloadSchema
from ..vector_db_interface import VectorDBInterface


class NeptuneAnalyticsAdapter(VectorDBInterface):
    name = "Neptune Analytics"

    VECTOR_NODE_IDENTIFIER = "COGNEE_VECTOR_NODE"
    COLLECITON_PREFIX = "VECTOR_COLLECTION_"

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

    async def prune(self):
        """
        Remove obsolete or unnecessary data from the database.
        """

        # result = self._client.query(f"MATCH (n:{self.VECTOR_NODE_IDENTIFIER}) RETURN n")

        # Run actual truncate
        result = self._client.query(f"MATCH (n:{self.VECTOR_NODE_IDENTIFIER}) DETACH DELETE n")
        pass


    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        """
        Delete specified data points from a collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete data
              points.
            - data_point_ids (list[str]): A list of IDs of the data points to delete.
        """
        query_string = (f"MATCH (n"
                        f":{self.VECTOR_NODE_IDENTIFIER} "
                        f":{self.COLLECITON_PREFIX}{collection_name}) "
                        f"WHERE id(n) IN {data_point_ids} "
                        f"DETACH DELETE n")
        print(query_string)
        result = self._client.query(query_string)
        pass


    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[PayloadSchema] = None,
    ):
        """
        Create a new collection with an optional payload schema.

        Parameters:
        -----------

            - collection_name (str): The name of the new collection to create.
            - payload_schema (Optional[PayloadSchema]): An optional schema for the payloads
              within this collection. (default None)
        """

        # In Neptune Analytics, node's label is being used to represent collection,
        # hence there is no-op to create collection when there is no element belongs to it.
        pass


    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: True if the collection exists, otherwise False.
        """
        query_string = (f"MATCH (n"
                        f":{self.VECTOR_NODE_IDENTIFIER} "
                        f":{self.COLLECITON_PREFIX}{collection_name}) "
                        f"RETURN true as collection_exist LIMIT 1")
        result = self._client.query(query_string)
        pass



    """ Data points """

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """
        Insert new data points into the specified collection.

        Parameters:
        -----------

            - collection_name (str): The name of the collection where data points will be added.
            - data_points (List[DataPoint]): A list of data points to be added to the
              collection.
        """

        for item in data_points:
            print(item)
            # query_string = (
            #         f"MERGE (n"
            #         f":{VECTOR_NODE_IDENTIFIER} "
            #         f":{COLLECITON_PREFIX}{collection_name}) "
            #         f"{{~id: '{node_id}'}} "
            #         f"WITH n "
            #         f"CALL neptune.algo.vectors.upsert('{node_id}', {embedding}) "
            #         f"YIELD node, embedding, success "
            #         f"RETURN node, embedding, success ")
            # result = self._client.query(query_string)
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
        pass

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
              results. (default False)
        """
        pass

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
        """
        pass

