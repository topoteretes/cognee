""" Neo4j Adapter for Graph Database"""
import json
import logging
from datetime import datetime
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager
from neo4j import AsyncSession
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = logging.getLogger(__name__)

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
            auth = (graph_database_username, graph_database_password)
        )

    async def close(self) -> None:
        await self.driver.close()

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
                return await result.data()
        except Neo4jError as error:
            logger.error("Neo4j query error: %s", error, exc_info = True)
            raise error

    async def graph(self):
        return await self.get_session()

    async def add_node(self, node_id: str, node_properties: Dict[str, Any] = None):
        node_id = node_id.replace(":", "_")

        # Serialize complex properties
        serialized_properties = {
            property_key: json.dumps(property_value)
                if isinstance(property_value, (dict, list))
                else property_value for property_key, property_value in node_properties.items()
        }

        if "name" not in serialized_properties:
            serialized_properties["name"] = node_id

        serialized_properties["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        serialized_properties["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        properties = ", ".join(f"{property_name}: ${property_name}" for property_name in serialized_properties.keys())

        query = (
            f"MERGE (n:`{node_id}` {{node_id: $node_id}}) "
            f"ON CREATE SET n += {{{properties}}} "
            f"RETURN  ID(n) AS internalId, n.node_id AS nodeId"
        )
        params = { "node_id": node_id, **serialized_properties }
        result = await self.query(query, params)

        await self.close()

        return result

    async def extract_node_description(self, id: str):
        query = f"""MATCH (n)-[r]->(m)
                    WHERE n.node_id = '{id}'
                    AND NOT m.node_id CONTAINS 'DOCUMENT'
                    AND m.layer_decomposition_uuid IS NOT NULL
                    RETURN m
                    """

        result = await self.query(query)

        await self.close()

        descriptions = []

        for node in result:
            # Assuming 'm' is a consistent key in your data structure
            attributes = node.get("m", {})

            # Ensure all required attributes are present
            if all(key in attributes for key in ["description", "unique_id", "layer_uuid", "layer_decomposition_uuid"]):
                descriptions.append({
                    "node_id": attributes["unique_id"],
                    "description": attributes["description"],
                    "layer_uuid": attributes["layer_uuid"],
                    "layer_decomposition_uuid": attributes["layer_decomposition_uuid"]
                })

        return descriptions



    async def extract_node(self, id: str):
        query= f"""MATCH(n) WHERE ID(n) = {id} RETURN n"""

        result = await self.query(query)
        await self.close()
        return result

    async def delete_node(self, id: str):
        node_id = id.replace(":", "_")

        query = "MATCH (n:{node_id} {node_id: node_id}) DETACH DELETE n"
        params = { "node_id": node_id }

        await self.query(query, params)
        await self.close()

    async def add_edge(self, from_node: str, to_node: str, relationship_name: str, edge_properties: Optional[Dict[str, Any]] = None):
        # Filter out None values and do not serialize; Neo4j can handle complex types like arrays directly
        filtered_properties = {
            property_name: property_value
                for property_name, property_value in edge_properties.items() if property_value is not None
        }
        from_node = from_node.replace(":", "_")
        to_node = to_node.replace(":", "_")

        # If there are no properties to add, simply create the relationship without properties
        if not filtered_properties:
            query = (
                f"MATCH (a:`{from_node}` {{node_id: $from_node}}), (b:`{to_node}` {{node_id: $to_node}}) "
                f"MERGE (a)-[r:`{relationship_name}`]->(b) "
                "RETURN r"
            )
            params = { "from_node": from_node, "to_node": to_node }
        else:
            # Prepare the SET clause to add properties to the relationship
            set_clause = ", ".join(f"r.{property_name} = ${property_name}" for property_name in filtered_properties.keys())
            query = (
                f"MATCH (a:`{from_node}` {{node_id: $from_node}}), (b:`{to_node}`  {{node_id: $to_node}}) "
                f"MERGE (a)-[r:`{relationship_name}`]->(b) "
                f"SET {set_clause} "
                "RETURN r"
            )

            params = { "from_node": from_node, "to_node": to_node, **filtered_properties }

        await self.query(query, params)
        await self.close()


    async def filter_nodes(self, search_criteria):
        query = f""" MATCH (d)
                WHERE d.node_id CONTAINS '{search_criteria}'
                RETURN d"""


        result = await self.query(query)
        await self.close()
        return result
