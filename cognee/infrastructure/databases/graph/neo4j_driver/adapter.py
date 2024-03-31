""" Neo4j Adapter for Graph Database"""
import json
import logging
from typing import Optional, Any, List, Dict
import asyncio

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from neo4j import AsyncGraphDatabase
from neo4j import AsyncSession
from neo4j.exceptions import Neo4jError
from contextlib import asynccontextmanager


class Neo4jAdapter(GraphDBInterface):
    def __init__(
            self, filename:str, graph_database_url: str, graph_database_username: str, graph_database_password: str, driver: Optional[Any] = None
    ):
        self.driver = driver or AsyncGraphDatabase.driver(
            graph_database_url, auth=(graph_database_username, graph_database_password)
        )

    async def close(self) -> None:
        await self.driver.close()

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        async with self.driver.session() as session:
            yield session

    async def query(
            self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            async with self.get_session() as session:
                result = await session.run(query, parameters=params)
                return await result.data()
        except Exception as e:
            logging.error(f"Neo4j query error: %s {e}")
            raise

    async def graph(self):
        return await self.get_session()

    async def add_node(self, id: str, **kwargs):
        """Asynchronously add a node to the graph if it doesn't already exist, with given properties."""

        # Serialize complex properties
        serialized_properties = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in kwargs.items()}

        properties = ', '.join(f'{k}: ${k}' for k in serialized_properties.keys())
        print("Adding node")
        query = (
            f"MERGE (n:Node {{id: $id}}) "
            f"ON CREATE SET n += {{{properties}}} "
            "RETURN n"
        )
        params = {'id': id, **serialized_properties}
        await self.query(query, params)
        await self.close()


    async def delete_node(self, id: str):
        """ Asynchronously delete a node from the graph if it exists."""
        query = "MATCH (n:Node {id: $id}) DETACH DELETE n"
        params = {'id': id}
        await self.query(query, params)
        await self.close()

    async def add_edge(self, from_node: str, to_node: str, relationship_type: str, **kwargs):
        """Asynchronously add an edge to the graph if it doesn't already exist, with given properties."""

        # Filter out None values and do not serialize; Neo4j can handle complex types like arrays directly
        filtered_properties = {k: v for k, v in kwargs.items() if v is not None}

        # If there are no properties to add, simply create the relationship without properties
        if not filtered_properties:
            query = (
                f"MATCH (a:Node {{id: $from_node}}), (b:Node {{id: $to_node}}) "
                f"MERGE (a)-[r:{relationship_type}]->(b) "
                "RETURN r"
            )
            params = {'from_node': from_node, 'to_node': to_node}
        else:
            # Prepare the SET clause to add properties to the relationship
            set_clause = ', '.join(f'r.{k} = ${k}' for k in filtered_properties.keys())
            query = (
                f"MATCH (a:Node {{id: $from_node}}), (b:Node {{id: $to_node}}) "
                f"MERGE (a)-[r:{relationship_type}]->(b) "
                f"SET {set_clause} "
                "RETURN r"
            )
            params = {'from_node': from_node, 'to_node': to_node, **filtered_properties}

        await self.query(query, params)
        await self.close()
    #
    # async def add_node(self, id: str, **kwargs) -> None:
    #     properties = ', '.join(f'{key}: ${key}' for key in kwargs)
    #     query = (
    #         f"MERGE (n:Node {{id: $id}}) "
    #         f"ON CREATE SET n += {{{properties}}} "
    #         f"RETURN n"
    #     )
    #     params = {'id': id, **kwargs}
    #     await self.query(query, params)
    #
    # # Implement the add_edge method from GraphDBInterface
    # async def add_edge(self, from_node: str, to_node: str, **kwargs) -> None:
    #     properties = ', '.join(f'{key}: ${key}' for key in kwargs)
    #     query = (
    #         "MATCH (a:Node), (b:Node) "
    #         "WHERE a.id = $from_id AND b.id = $to_id "
    #         f"MERGE (a)-[r:RELATES {{{properties}}}]->(b) "
    #         "RETURN r"
    #     )
    #     params = {'from_id': from_node, 'to_id': to_node, **kwargs}
    #     await self.query(query, params)