import asyncio

# from datetime import datetime
import json
from textwrap import dedent
from uuid import UUID
from webbrowser import Error

from falkordb import FalkorDB

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.vector.embeddings import EmbeddingEngine
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.engine import DataPoint


class IndexSchema(DataPoint):
    text: str

    metadata: dict = {"index_fields": ["text"]}


class FalkorDBAdapter(VectorDBInterface, GraphDBInterface):
    def __init__(
        self,
        database_url: str,
        database_port: int,
        embedding_engine=EmbeddingEngine,
    ):
        self.driver = FalkorDB(
            host=database_url,
            port=database_port,
        )
        self.embedding_engine = embedding_engine
        self.graph_name = "cognee_graph"

    def query(self, query: str, params: dict = {}):
        graph = self.driver.select_graph(self.graph_name)

        try:
            result = graph.query(query, params)
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            raise e

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def stringify_properties(self, properties: dict) -> str:
        def parse_value(value):
            if type(value) is UUID:
                return f"'{str(value)}'"
            if type(value) is int or type(value) is float:
                return value
            if (
                type(value) is list
                and type(value[0]) is float
                and len(value) == self.embedding_engine.get_vector_size()
            ):
                return f"'vecf32({value})'"
            # if type(value) is datetime:
            #     return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
            if type(value) is dict:
                return f"'{json.dumps(value)}'"
            return f"'{value}'"

        return ",".join([f"{key}:{parse_value(value)}" for key, value in properties.items()])

    async def create_data_point_query(self, data_point: DataPoint, vectorized_values: dict):
        node_label = type(data_point).__name__
        property_names = DataPoint.get_embeddable_property_names(data_point)

        node_properties = await self.stringify_properties(
            {
                **data_point.model_dump(),
                **(
                    {
                        property_names[index]: (
                            vectorized_values[index]
                            if index < len(vectorized_values)
                            else getattr(data_point, property_name, None)
                        )
                        for index, property_name in enumerate(property_names)
                    }
                ),
            }
        )

        return dedent(
            f"""
            MERGE (node:{node_label} {{id: '{str(data_point.id)}'}})
            ON CREATE SET node += ({{{node_properties}}}), node.updated_at = timestamp()
            ON MATCH SET node += ({{{node_properties}}}), node.updated_at = timestamp()
        """
        ).strip()

    async def create_edge_query(self, edge: tuple[str, str, str, dict]) -> str:
        properties = await self.stringify_properties(edge[3])
        properties = f"{{{properties}}}"

        return dedent(
            f"""
            MERGE (source {{id:'{edge[0]}'}})
            MERGE (target {{id: '{edge[1]}'}})
            MERGE (source)-[edge:{edge[2]} {properties}]->(target)
            ON MATCH SET edge.updated_at = timestamp()
            ON CREATE SET edge.updated_at = timestamp()
        """
        ).strip()

    async def create_collection(self, collection_name: str):
        pass

    async def has_collection(self, collection_name: str) -> bool:
        collections = self.driver.list_graphs()

        return collection_name in collections

    async def create_data_points(self, data_points: list[DataPoint]):
        embeddable_values = []
        vector_map = {}

        for data_point in data_points:
            property_names = DataPoint.get_embeddable_property_names(data_point)
            key = str(data_point.id)
            vector_map[key] = {}

            for property_name in property_names:
                property_value = getattr(data_point, property_name, None)

                if property_value is not None:
                    vector_map[key][property_name] = len(embeddable_values)
                    embeddable_values.append(property_value)
                else:
                    vector_map[key][property_name] = None

        vectorized_values = await self.embed_data(embeddable_values)

        queries = [
            await self.create_data_point_query(
                data_point,
                [
                    vectorized_values[vector_map[str(data_point.id)][property_name]]
                    if vector_map[str(data_point.id)][property_name] is not None
                    else None
                    for property_name in DataPoint.get_embeddable_property_names(data_point)
                ],
            )
            for data_point in data_points
        ]

        for query in queries:
            self.query(query)

    async def create_vector_index(self, index_name: str, index_property_name: str):
        graph = self.driver.select_graph(self.graph_name)

        if not self.has_vector_index(graph, index_name, index_property_name):
            graph.create_node_vector_index(
                index_name, index_property_name, dim=self.embedding_engine.get_vector_size()
            )

    def has_vector_index(self, graph, index_name: str, index_property_name: str) -> bool:
        try:
            indices = graph.list_indices()

            return any(
                [
                    (index[0] == index_name and index_property_name in index[1])
                    for index in indices.result_set
                ]
            )
        except Error as e:
            print(e)
            return False

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        pass

    async def add_node(self, node: DataPoint):
        await self.create_data_points([node])

    async def add_nodes(self, nodes: list[DataPoint]):
        await self.create_data_points(nodes)

    async def add_edge(self, edge: tuple[str, str, str, dict]):
        query = await self.create_edge_query(edge)

        self.query(query)

    async def add_edges(self, edges: list[tuple[str, str, str, dict]]):
        queries = [await self.create_edge_query(edge) for edge in edges]

        for query in queries:
            self.query(query)

    async def has_edges(self, edges):
        query = dedent(
            """
            UNWIND $edges AS edge
            MATCH (a)-[r]->(b)
            WHERE id(a) = edge.from_node AND id(b) = edge.to_node AND type(r) = edge.relationship_name
            RETURN edge.from_node AS from_node, edge.to_node AS to_node, edge.relationship_name AS relationship_name, count(r) > 0 AS edge_exists
        """
        ).strip()

        params = {
            "edges": [
                {
                    "from_node": str(edge[0]),
                    "to_node": str(edge[1]),
                    "relationship_name": edge[2],
                }
                for edge in edges
            ],
        }

        results = self.query(query, params).result_set

        return [result["edge_exists"] for result in results]

    async def retrieve(self, data_point_ids: list[UUID]):
        result = self.query(
            "MATCH (node) WHERE node.id IN $node_ids RETURN node",
            {
                "node_ids": [str(data_point) for data_point in data_point_ids],
            },
        )
        return result.result_set

    async def extract_node(self, data_point_id: UUID):
        result = await self.retrieve([data_point_id])
        result = result[0][0] if len(result[0]) > 0 else None
        return result.properties if result else None

    async def extract_nodes(self, data_point_ids: list[UUID]):
        return await self.retrieve(data_point_ids)

    async def get_connections(self, node_id: UUID) -> list:
        predecessors_query = """
        MATCH (node)<-[relation]-(neighbour)
        WHERE node.id = $node_id
        RETURN neighbour, relation, node
        """
        successors_query = """
        MATCH (node)-[relation]->(neighbour)
        WHERE node.id = $node_id
        RETURN node, relation, neighbour
        """

        predecessors, successors = await asyncio.gather(
            self.query(predecessors_query, dict(node_id=node_id)),
            self.query(successors_query, dict(node_id=node_id)),
        )

        connections = []

        for neighbour in predecessors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        for neighbour in successors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        return connections

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: list[float] = None,
        limit: int = 10,
        with_vector: bool = False,
    ):
        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embed_data([query_text]))[0]

        [label, attribute_name] = collection_name.split(".")

        query = dedent(
            f"""
            CALL db.idx.vector.queryNodes(
                '{label}',
                '{attribute_name}',
                {limit},
                vecf32({query_vector})
            ) YIELD node, score
        """
        ).strip()

        result = self.query(query)

        return result.result_set

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        query_vectors = await self.embedding_engine.embed_text(query_texts)

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

    async def get_graph_data(self):
        query = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"

        result = self.query(query)

        nodes = [
            (
                record[2]["id"],
                record[2],
            )
            for record in result.result_set
        ]

        query = """
        MATCH (n)-[r]->(m)
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result = self.query(query)
        edges = [
            (
                record[3]["source_node_id"],
                record[3]["target_node_id"],
                record[2],
                record[3],
            )
            for record in result.result_set
        ]

        return (nodes, edges)

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        return self.query(
            "MATCH (node) WHERE node.id IN $node_ids DETACH DELETE node",
            {
                "node_ids": [str(data_point) for data_point in data_point_ids],
            },
        )

    async def delete_node(self, collection_name: str, data_point_id: str):
        return await self.delete_data_points([data_point_id])

    async def delete_nodes(self, collection_name: str, data_point_ids: list[str]):
        self.delete_data_points(data_point_ids)

    async def delete_graph(self):
        try:
            graph = self.driver.select_graph(self.graph_name)

            indices = graph.list_indices()
            for index in indices.result_set:
                for field in index[1]:
                    graph.drop_node_vector_index(index[0], field)

            graph.delete()
        except Exception as e:
            print(f"Error deleting graph: {e}")

    async def prune(self):
        await self.delete_graph()
