""" Neo4j Adapter for Graph Database"""

import logging
from typing import Optional, Any, List, Dict

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


    async def add_node(self, id: str, **kwargs):
        """Asynchronously add a node to the graph if it doesn't already exist, with given properties."""
        properties = ', '.join(f'{k}: ${k}' for k in kwargs.keys())
        query = (
            f"MERGE (n:Node {{id: $id}}) "
            f"ON CREATE SET n += {{{properties}}} "
            "RETURN n"
        )
        params = {'id': id, **kwargs}
        await self.query(query, params)


    async def delete_node(self, id: str):
        """ Asynchronously delete a node from the graph if it exists."""
        query = "MATCH (n:Node {id: $id}) DETACH DELETE n"
        params = {'id': id}
        await self.query(query, params)


    async def add_edge(self, from_node: str, to_node: str, **kwargs):
        """Asynchronously add a edges to the graph if it doesn't already exist, with given properties."""
        properties = ', '.join(f'{k}: ${k}' for k in kwargs.keys())
        query = (
            "MATCH (a:Node {id: $from_node}), (b:Node {id: $to_node}) "
            f"MERGE (a)-[r:RELATES {{{properties}}}]->(b) "
            "RETURN r"
        )
        params = {'from_node': from_node, 'to_node': to_node, **kwargs}
        await self.query(query, params)
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