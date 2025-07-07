from typing import List, Optional
from langchain_aws import NeptuneAnalyticsGraph, NeptuneGraph
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import get_own_properties
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.PayloadSchema import PayloadSchema
from ..vector_db_interface import VectorDBInterface

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
    COLLECTION_PREFIX = "VECTOR_COLLECTION_"

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
        Check if a specified collection exists,
        by issuing an Opencypher query to check any vector node has the collection label.

        Parameters:
        -----------
            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: True if the collection exists, otherwise False.
        """
        query_string = (f"MATCH (n"
                        f":{self.VECTOR_NODE_IDENTIFIER} "
                        f":{self.COLLECTION_PREFIX}{collection_name}) "
                        f"RETURN COUNT(n) > 0 as collection_exist LIMIT 1")
        result = self._client.query(query_string)
        return result[0]['collection_exist']


    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[PayloadSchema] = None,
    ):
        """
        In Neptune Analytics, node's label is being used to represent collection,
        hence this method will be no-op,
        has_collection() will return True when one or more vector being inserted,
        and False otherwise.

        Parameters:
        -----------

            - collection_name (str): The name of the new collection to create.
            - payload_schema (Optional[PayloadSchema]): An optional schema for the payloads
              within this collection. (default None)
        """
        pass




    """ Node operations """

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
        query_string = (f"MATCH (n"
                        f":{self.VECTOR_NODE_IDENTIFIER} "
                        f":{self.COLLECTION_PREFIX}{collection_name}) "
                        f"WHERE id(n) IN {data_point_ids} "
                        f"DETACH DELETE n")
        self._client.query(query_string)
        pass

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
        for data_point in data_points:


            node_id = data_point.id
            # Generate embedding
            text_content = DataPoint.get_embeddable_data(data_point)
            data_vectors = await self.embedding_engine.embed_text([text_content])

            # Fetch properties
            # properties = get_own_properties(data_point)
            properties = {"test": "value"}

            params = {
                "node_id": node_id,
                "properties": properties
            }


            # Composite the query and send
            query_string = (
                    f"CREATE (n"
                    f":{self.VECTOR_NODE_IDENTIFIER} "
                    f":{self.COLLECTION_PREFIX}{collection_name} "
                    f" $properties ) "
                    # f"{{`~id`: '{node_id}'}}) "
                    f"WITH n "
                    f"CALL neptune.algo.vectors.upsert('{node_id}', {data_vectors}) "
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
        result_set = []

        # Do the fetch for each node
        for node_id in data_point_ids:
            # Composite query
            query_string = (f"MATCH( n "
                            f":{self.VECTOR_NODE_IDENTIFIER} "
                            f":{self.COLLECTION_PREFIX}{collection_name} "
                            f"{{`~id`: '{node_id}'}}) "
                            f"CALL neptune.algo.vectors.get(n) "
                            f"YIELD embedding RETURN id(n), embedding")
            result = self._client.query(query_string)
            result_set.append(result)
        return result_set


    """ Graph operation """

    async def prune(self):
        """
        Remove obsolete or unnecessary data from the database.
        """
        # Run actual truncate
        self._client.query(f"MATCH (n:{self.VECTOR_NODE_IDENTIFIER}) DETACH DELETE n")
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

