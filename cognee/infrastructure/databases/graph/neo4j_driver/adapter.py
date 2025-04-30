#

"""Neo4j Adapter for Graph Database"""

import json
from cognee.shared.logging_utils import get_logger, ERROR
import asyncio
from textwrap import dedent
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager
from uuid import UUID
from neo4j import AsyncSession
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.modules.storage.utils import JSONEncoder
from .neo4j_metrics_utils import (
    get_avg_clustering,
    get_edge_density,
    get_num_connected_components,
    get_shortest_path_lengths,
    get_size_of_connected_components,
    count_self_loops,
)

logger = get_logger("Neo4jAdapter", level=ERROR)


class Neo4jAdapter(GraphDBInterface):
    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: str,
        graph_database_password: str,
        driver: Optional[Any] = None,
    ):
        self.driver = driver or AsyncGraphDatabase.driver(
            graph_database_url,
            auth=(graph_database_username, graph_database_password),
            max_connection_lifetime=120,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        async with self.driver.session() as session:
            yield session

    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            async with self.get_session() as session:
                result = await session.run(query, parameters=params)
                data = await result.data()
                return data
        except Neo4jError as error:
            logger.error("Neo4j query error: %s", error, exc_info=True)
            raise error

    async def has_node(self, node_id: str) -> bool:
        results = self.query(
            """
                MATCH (n)
                WHERE n.id = $node_id
                RETURN COUNT(n) > 0 AS node_exists
            """,
            {"node_id": node_id},
        )
        return results[0]["node_exists"] if len(results) > 0 else False

    async def add_node(self, node: DataPoint):
        serialized_properties = self.serialize_properties(node.model_dump())

        query = dedent(
            """MERGE (node {id: $node_id})
                ON CREATE SET node += $properties, node.updated_at = timestamp()
                ON MATCH SET node += $properties, node.updated_at = timestamp()
                WITH node, $node_label AS label
                CALL apoc.create.addLabels(node, [label]) YIELD node AS labeledNode
                RETURN ID(labeledNode) AS internal_id, labeledNode.id AS nodeId"""
        )

        params = {
            "node_id": str(node.id),
            "node_label": type(node).__name__,
            "properties": serialized_properties,
        }

        return await self.query(query, params)

    @record_graph_changes
    async def add_nodes(self, nodes: list[DataPoint]) -> None:
        query = """
        UNWIND $nodes AS node
        MERGE (n {id: node.node_id})
        ON CREATE SET n += node.properties, n.updated_at = timestamp()
        ON MATCH SET n += node.properties, n.updated_at = timestamp()
        WITH n, node.label AS label
        CALL apoc.create.addLabels(n, [label]) YIELD node AS labeledNode
        RETURN ID(labeledNode) AS internal_id, labeledNode.id AS nodeId
        """

        nodes = [
            {
                "node_id": str(node.id),
                "label": type(node).__name__,
                "properties": self.serialize_properties(node.model_dump()),
            }
            for node in nodes
        ]

        results = await self.query(query, dict(nodes=nodes))
        return results

    async def extract_node(self, node_id: str):
        results = await self.extract_nodes([node_id])

        return results[0] if len(results) > 0 else None

    async def extract_nodes(self, node_ids: List[str]):
        query = """
        UNWIND $node_ids AS id
        MATCH (node {id: id})
        RETURN node"""

        params = {"node_ids": node_ids}

        results = await self.query(query, params)

        return [result["node"] for result in results]

    async def delete_node(self, node_id: str):
        query = "MATCH (node {id: $node_id}) DETACH DELETE node"
        params = {"node_id": node_id}

        return await self.query(query, params)

    async def delete_nodes(self, node_ids: list[str]) -> None:
        query = """
        UNWIND $node_ids AS id
        MATCH (node {id: id})
        DETACH DELETE node"""

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def has_edge(self, from_node: UUID, to_node: UUID, edge_label: str) -> bool:
        query = """
            MATCH (from_node)-[relationship]->(to_node)
            WHERE from_node.id = $from_node_id AND to_node.id = $to_node_id AND type(relationship) = $edge_label
            RETURN COUNT(relationship) > 0 AS edge_exists
        """

        params = {
            "from_node_id": str(from_node),
            "to_node_id": str(to_node),
            "edge_label": edge_label,
        }

        edge_exists = await self.query(query, params)
        return edge_exists

    async def has_edges(self, edges):
        query = """
            UNWIND $edges AS edge
            MATCH (a)-[r]->(b)
            WHERE id(a) = edge.from_node AND id(b) = edge.to_node AND type(r) = edge.relationship_name
            RETURN edge.from_node AS from_node, edge.to_node AS to_node, edge.relationship_name AS relationship_name, count(r) > 0 AS edge_exists
        """

        try:
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

            results = await self.query(query, params)
            return [result["edge_exists"] for result in results]
        except Neo4jError as error:
            logger.error("Neo4j query error: %s", error, exc_info=True)
            raise error

    async def add_edge(
        self,
        from_node: UUID,
        to_node: UUID,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = {},
    ):
        serialized_properties = self.serialize_properties(edge_properties)

        query = dedent(
            f"""\
            MATCH (from_node {{id: $from_node}}),
                  (to_node {{id: $to_node}})
            MERGE (from_node)-[r:{relationship_name}]->(to_node)
            ON CREATE SET r += $properties, r.updated_at = timestamp()
            ON MATCH SET r += $properties, r.updated_at = timestamp()
            RETURN r
            """
        )

        params = {
            "from_node": str(from_node),
            "to_node": str(to_node),
            "relationship_name": relationship_name,
            "properties": serialized_properties,
        }

        return await self.query(query, params)

    @record_graph_changes
    async def add_edges(self, edges: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        query = """
            UNWIND $edges AS edge
            MATCH (from_node {id: edge.from_node})
            MATCH (to_node {id: edge.to_node})
            CALL apoc.merge.relationship(
                from_node,
                edge.relationship_name,
                {
                    source_node_id: edge.from_node,
                    target_node_id: edge.to_node
                },
                edge.properties,
                to_node
            ) YIELD rel
            RETURN rel"""

        edges = [
            {
                "from_node": str(edge[0]),
                "to_node": str(edge[1]),
                "relationship_name": edge[2],
                "properties": {
                    **(edge[3] if edge[3] else {}),
                    "source_node_id": str(edge[0]),
                    "target_node_id": str(edge[1]),
                },
            }
            for edge in edges
        ]

        try:
            results = await self.query(query, dict(edges=edges))
            return results
        except Neo4jError as error:
            logger.error("Neo4j query error: %s", error, exc_info=True)
            raise error

    async def get_edges(self, node_id: str):
        query = """
        MATCH (n {id: $node_id})-[r]-(m)
        RETURN n, r, m
        """

        results = await self.query(query, dict(node_id=node_id))

        return [
            (result["n"]["id"], result["m"]["id"], {"relationship_name": result["r"][1]})
            for result in results
        ]

    async def get_disconnected_nodes(self) -> list[str]:
        # return await self.query(
        #     "MATCH (node) WHERE NOT (node)<-[:*]-() RETURN node.id as id",
        # )
        query = """
        // Step 1: Collect all nodes
        MATCH (n)
        WITH COLLECT(n) AS nodes

        // Step 2: Find all connected components
        WITH nodes
        CALL {
          WITH nodes
          UNWIND nodes AS startNode
          MATCH path = (startNode)-[*]-(connectedNode)
          WITH startNode, COLLECT(DISTINCT connectedNode) AS component
          RETURN component
        }

        // Step 3: Aggregate components
        WITH COLLECT(component) AS components

        // Step 4: Identify the largest connected component
        UNWIND components AS component
        WITH component
        ORDER BY SIZE(component) DESC
        LIMIT 1
        WITH component AS largestComponent

        // Step 5: Find nodes not in the largest connected component
        MATCH (n)
        WHERE NOT n IN largestComponent
        RETURN COLLECT(ID(n)) AS ids
        """

        results = await self.query(query)
        return results[0]["ids"] if len(results) > 0 else []

    async def get_predecessors(self, node_id: str, edge_label: str = None) -> list[str]:
        if edge_label is not None:
            query = """
            MATCH (node)<-[r]-(predecessor)
            WHERE node.id = $node_id AND type(r) = $edge_label
            RETURN predecessor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                    edge_label=edge_label,
                ),
            )

            return [result["predecessor"] for result in results]
        else:
            query = """
            MATCH (node)<-[r]-(predecessor)
            WHERE node.id = $node_id
            RETURN predecessor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                ),
            )

            return [result["predecessor"] for result in results]

    async def get_successors(self, node_id: str, edge_label: str = None) -> list[str]:
        if edge_label is not None:
            query = """
            MATCH (node)-[r]->(successor)
            WHERE node.id = $node_id AND type(r) = $edge_label
            RETURN successor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                    edge_label=edge_label,
                ),
            )

            return [result["successor"] for result in results]
        else:
            query = """
            MATCH (node)-[r]->(successor)
            WHERE node.id = $node_id
            RETURN successor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                ),
            )

            return [result["successor"] for result in results]

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all neighboring nodes."""
        return await self.get_neighbours(node_id)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID."""
        query = """
        MATCH (node {id: $node_id})
        RETURN node
        """
        results = await self.query(query, {"node_id": node_id})
        return results[0]["node"] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple nodes by their IDs."""
        query = """
        UNWIND $node_ids AS id
        MATCH (node {id: id})
        RETURN node
        """
        results = await self.query(query, {"node_ids": node_ids})
        return [result["node"] for result in results]

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
            self.query(predecessors_query, dict(node_id=str(node_id))),
            self.query(successors_query, dict(node_id=str(node_id))),
        )

        connections = []

        for neighbour in predecessors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        for neighbour in successors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        return connections

    async def remove_connection_to_predecessors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node:`{id}`)-[r:{edge_label}]->(predecessor)
        DELETE r;
        """

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def remove_connection_to_successors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node:`{id}`)<-[r:{edge_label}]-(successor)
        DELETE r;
        """

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def delete_graph(self):
        query = """MATCH (node)
                DETACH DELETE node;"""

        return await self.query(query)

    def serialize_properties(self, properties=dict()):
        serialized_properties = {}

        for property_key, property_value in properties.items():
            if isinstance(property_value, UUID):
                serialized_properties[property_key] = str(property_value)
                continue

            if isinstance(property_value, dict):
                serialized_properties[property_key] = json.dumps(property_value, cls=JSONEncoder)
                continue

            serialized_properties[property_key] = property_value

        return serialized_properties

    async def get_model_independent_graph_data(self):
        query_nodes = "MATCH (n) RETURN collect(n) AS nodes"
        nodes = await self.query(query_nodes)

        query_edges = "MATCH (n)-[r]->(m) RETURN collect([n, r, m]) AS elements"
        edges = await self.query(query_edges)

        return (nodes, edges)

    async def get_graph_data(self):
        query = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"

        result = await self.query(query)

        nodes = [
            (
                record["properties"]["id"],
                record["properties"],
            )
            for record in result
        ]

        query = """
        MATCH (n)-[r]->(m)
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result = await self.query(query)
        edges = [
            (
                record["properties"]["source_node_id"],
                record["properties"]["target_node_id"],
                record["type"],
                record["properties"],
            )
            for record in result
        ]

        return (nodes, edges)

    async def get_filtered_graph_data(self, attribute_filters):
        """
        Fetches nodes and relationships filtered by specified attribute values.

        Args:
            attribute_filters (list of dict): A list of dictionaries where keys are attributes and values are lists of values to filter on.
                                              Example: [{"community": ["1", "2"]}]

        Returns:
            tuple: A tuple containing two lists: nodes and edges.
        """
        where_clauses = []
        for attribute, values in attribute_filters[0].items():
            values_str = ", ".join(
                f"'{value}'" if isinstance(value, str) else str(value) for value in values
            )
            where_clauses.append(f"n.{attribute} IN [{values_str}]")

        where_clause = " AND ".join(where_clauses)

        query_nodes = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties
        """
        result_nodes = await self.query(query_nodes)

        nodes = [
            (
                record["id"],
                record["properties"],
            )
            for record in result_nodes
        ]

        query_edges = f"""
        MATCH (n)-[r]->(m)
        WHERE {where_clause} AND {where_clause.replace("n.", "m.")}
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result_edges = await self.query(query_edges)

        edges = [
            (
                record["source"],
                record["target"],
                record["type"],
                record["properties"],
            )
            for record in result_edges
        ]

        return (nodes, edges)

    async def graph_exists(self, graph_name="myGraph"):
        query = "CALL gds.graph.list() YIELD graphName RETURN collect(graphName) AS graphNames;"
        result = await self.query(query)
        graph_names = result[0]["graphNames"] if result else []
        return graph_name in graph_names

    async def get_node_labels_string(self):
        node_labels_query = "CALL db.labels() YIELD label RETURN collect(label) AS labels;"
        node_labels_result = await self.query(node_labels_query)
        node_labels = node_labels_result[0]["labels"] if node_labels_result else []

        if not node_labels:
            raise ValueError("No node labels found in the database")

        node_labels_str = "[" + ", ".join(f"'{label}'" for label in node_labels) + "]"
        return node_labels_str

    async def get_relationship_labels_string(self):
        relationship_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS relationships;"
        relationship_types_result = await self.query(relationship_types_query)
        relationship_types = (
            relationship_types_result[0]["relationships"] if relationship_types_result else []
        )

        if not relationship_types:
            raise ValueError("No relationship types found in the database.")

        relationship_types_undirected_str = (
            "{"
            + ", ".join(f"{rel}" + ": {orientation: 'UNDIRECTED'}" for rel in relationship_types)
            + "}"
        )
        return relationship_types_undirected_str

    async def project_entire_graph(self, graph_name="myGraph"):
        """
        Projects all node labels and all relationship types into an undirected in-memory GDS graph.
        """
        if await self.graph_exists(graph_name):
            return

        node_labels_str = await self.get_node_labels_string()
        relationship_types_undirected_str = await self.get_relationship_labels_string()

        query = f"""
        CALL gds.graph.project(
            '{graph_name}',
            {node_labels_str},
            {relationship_types_undirected_str}
        ) YIELD graphName;
        """

        await self.query(query)

    async def drop_graph(self, graph_name="myGraph"):
        if await self.graph_exists(graph_name):
            drop_query = f"CALL gds.graph.drop('{graph_name}');"
            await self.query(drop_query)

    async def get_graph_metrics(self, include_optional=False):
        """For the definition of these metrics, please refer to
        https://docs.cognee.ai/core_concepts/graph_generation/descriptive_metrics"""

        nodes, edges = await self.get_model_independent_graph_data()
        graph_name = "myGraph"
        await self.drop_graph(graph_name)
        await self.project_entire_graph(graph_name)

        num_nodes = len(nodes[0]["nodes"])
        num_edges = len(edges[0]["elements"])

        mandatory_metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
            "edge_density": await get_edge_density(self),
            "num_connected_components": await get_num_connected_components(self, graph_name),
            "sizes_of_connected_components": await get_size_of_connected_components(
                self, graph_name
            ),
        }

        if include_optional:
            shortest_path_lengths = await get_shortest_path_lengths(self, graph_name)
            optional_metrics = {
                "num_selfloops": await count_self_loops(self),
                "diameter": max(shortest_path_lengths) if shortest_path_lengths else -1,
                "avg_shortest_path_length": sum(shortest_path_lengths) / len(shortest_path_lengths)
                if shortest_path_lengths
                else -1,
                "avg_clustering": await get_avg_clustering(self, graph_name),
            }
        else:
            optional_metrics = {
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }

        return mandatory_metrics | optional_metrics

    async def get_document_subgraph(self, content_hash: str):
        query = """
        MATCH (doc)
        WHERE (doc:TextDocument OR doc:PdfDocument)
        AND doc.name = 'text_' + $content_hash

        OPTIONAL MATCH (doc)<-[:is_part_of]-(chunk:DocumentChunk)
        OPTIONAL MATCH (chunk)-[:contains]->(entity:Entity)
        WHERE NOT EXISTS {
            MATCH (entity)<-[:contains]-(otherChunk:DocumentChunk)-[:is_part_of]->(otherDoc)
            WHERE (otherDoc:TextDocument OR otherDoc:PdfDocument)
            AND otherDoc.id <> doc.id
        }
        OPTIONAL MATCH (chunk)<-[:made_from]-(made_node:TextSummary)
        OPTIONAL MATCH (entity)-[:is_a]->(type:EntityType)
        WHERE NOT EXISTS {
            MATCH (type)<-[:is_a]-(otherEntity:Entity)<-[:contains]-(otherChunk:DocumentChunk)-[:is_part_of]->(otherDoc)
            WHERE (otherDoc:TextDocument OR otherDoc:PdfDocument)
            AND otherDoc.id <> doc.id
        }

        RETURN
            collect(DISTINCT doc) as document,
            collect(DISTINCT chunk) as chunks,
            collect(DISTINCT entity) as orphan_entities,
            collect(DISTINCT made_node) as made_from_nodes,
            collect(DISTINCT type) as orphan_types
        """
        result = await self.query(query, {"content_hash": content_hash})
        return result[0] if result else None

    async def get_degree_one_nodes(self, node_type: str):
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        query = f"""
        MATCH (n:{node_type})
        WHERE COUNT {{ MATCH (n)--() }} = 1
        RETURN n
        """
        result = await self.query(query)
        return [record["n"] for record in result] if result else []
