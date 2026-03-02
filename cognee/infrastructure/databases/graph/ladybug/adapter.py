"""Adapter for Ladybug graph database."""

import os
import json
import asyncio
import tempfile
from uuid import UUID, uuid5, NAMESPACE_OID
import real_ladybug as lb
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Union, Optional, Tuple, Type

from cognee.exceptions import CogneeValidationError
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.utils.run_sync import run_sync
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.engine.utils.generate_timestamp_datapoint import date_to_int
from cognee.tasks.temporal_graph.models import Timestamp
from cognee.infrastructure.databases.cache.config import get_cache_config

logger = get_logger()

cache_config = get_cache_config()
if cache_config.shared_kuzu_lock:
    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine


class LadybugAdapter(GraphDBInterface):
    """
    Adapter for Ladybug graph database operations with improved consistency and async support.

    This class facilitates operations for working with the Ladybug graph database, supporting
    both direct database queries and a structured asynchronous interface for node and edge
    management. It contains methods for querying, adding, and deleting nodes and edges as
    well as for graph metrics and data extraction.
    """

    def __init__(self, db_path: str):
        """Initialize Ladybug database connection and schema."""
        self.open_connections = 0
        self._is_closed = False
        self.db_path = db_path
        self.db: Optional[lb.Database] = None
        self.connection: Optional[lb.Connection] = None
        if cache_config.shared_kuzu_lock:
            self.redis_lock = get_cache_engine(
                lock_key="ladybug-lock-" + str(uuid5(NAMESPACE_OID, db_path))
            )
        else:
            self.executor = ThreadPoolExecutor()
            self._initialize_connection()
        self.LADYBUG_ASYNC_LOCK = asyncio.Lock()
        self._connection_change_lock = asyncio.Lock()

    def _initialize_connection(self) -> None:
        """Initialize the Ladybug database connection and schema."""

        def _install_json_extension():
            """
            Function handles installing of the json extension for the current Ladybug version.
            This has to be done with an empty graph db before connecting to an existing database otherwise
            missing json extension errors will be raised.
            """
            try:
                with tempfile.NamedTemporaryFile(mode="w", delete=True) as temp_file:
                    temp_graph_file = temp_file.name
                    tmp_db = lb.Database(
                        temp_graph_file,
                        buffer_pool_size=2048 * 1024 * 1024,
                        max_db_size=4096 * 1024 * 1024,
                    )
                    tmp_db.init_database()
                    connection = lb.Connection(tmp_db)
                    connection.execute("INSTALL JSON;")
            except Exception as e:
                logger.info(f"JSON extension already installed or not needed: {e}")

        _install_json_extension()

        try:
            if "s3://" in self.db_path:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    self.temp_graph_file = temp_file.name

                run_sync(self.pull_from_s3())

                self.db = lb.Database(
                    self.temp_graph_file,
                    buffer_pool_size=2048 * 1024 * 1024,
                    max_db_size=4096 * 1024 * 1024,
                )
            else:
                db_dir = os.path.dirname(self.db_path)

                if not db_dir:
                    abs_path = os.path.abspath(self.db_path)
                    db_dir = os.path.dirname(abs_path)

                file_storage = get_file_storage(db_dir)

                run_sync(file_storage.ensure_directory_exists())

                try:
                    self.db = lb.Database(
                        self.db_path,
                        buffer_pool_size=2048 * 1024 * 1024,
                        max_db_size=4096 * 1024 * 1024,
                    )
                except RuntimeError:
                    from .ladybug_migrate import read_ladybug_storage_version

                    ladybug_db_version = read_ladybug_storage_version(self.db_path)
                    if (
                        ladybug_db_version == "0.9.0" or ladybug_db_version == "0.8.2"
                    ) and ladybug_db_version != lb.__version__:
                        from .ladybug_migrate import ladybug_migration

                        ladybug_migration(
                            new_db=self.db_path + "_new",
                            old_db=self.db_path,
                            new_version=lb.__version__,
                            old_version=ladybug_db_version,
                            overwrite=True,
                        )

                    self.db = lb.Database(
                        self.db_path,
                        buffer_pool_size=2048 * 1024 * 1024,
                        max_db_size=4096 * 1024 * 1024,
                    )

            self.db.init_database()
            self.connection = lb.Connection(self.db)

            try:
                self.connection.execute("LOAD EXTENSION JSON;")
                logger.info("Loaded JSON extension")
            except Exception as e:
                logger.info(f"JSON extension already loaded or unavailable: {e}")

            self.connection.execute("""
                CREATE NODE TABLE IF NOT EXISTS Node(
                    id STRING PRIMARY KEY,
                    name STRING,
                    type STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    properties STRING
                )
            """)
            self.connection.execute("""
                CREATE REL TABLE IF NOT EXISTS EDGE(
                    FROM Node TO Node,
                    relationship_name STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    properties STRING
                )
            """)
            logger.debug("Ladybug database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ladybug database: {e}")
            raise e

    async def push_to_s3(self) -> None:
        if os.getenv("STORAGE_BACKEND", "").lower() == "s3" and hasattr(self, "temp_graph_file"):
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

            s3_file_storage = S3FileStorage("")

            if self.connection:
                async with self.LADYBUG_ASYNC_LOCK:
                    self.connection.execute("CHECKPOINT;")

            s3_file_storage.s3.put(self.temp_graph_file, self.db_path, recursive=True)

    async def pull_from_s3(self) -> None:
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        s3_file_storage = S3FileStorage("")
        try:
            s3_file_storage.s3.get(self.db_path, self.temp_graph_file, recursive=True)
        except FileNotFoundError:
            logger.warning(f"Ladybug S3 storage file not found: {self.db_path}")

    async def is_empty(self) -> bool:
        query = """
        MATCH (n)
        RETURN true
        LIMIT 1;
        """
        query_result = await self.query(query)
        return len(query_result) == 0

    async def query(self, query: str, params: Optional[dict] = None) -> List[Tuple]:
        """
        Execute a Ladybug query asynchronously with automatic reconnection.
        """
        loop = asyncio.get_running_loop()
        params = params or {}

        def blocking_query():
            lock_acquired = False
            try:
                if cache_config.shared_kuzu_lock:
                    self.redis_lock.acquire_lock()
                    lock_acquired = True
                if not self.connection:
                    logger.info("Reconnecting to Ladybug database...")
                    self._initialize_connection()

                result = self.connection.execute(query, params)
                rows = []

                while result.has_next():
                    row = result.get_next()
                    processed_rows = []
                    for val in row:
                        if hasattr(val, "as_py"):
                            val = val.as_py()
                        processed_rows.append(val)
                    rows.append(tuple(processed_rows))

                return rows
            except Exception as e:
                logger.error(f"Query execution failed: {str(e)}")
                raise
            finally:
                if cache_config.shared_kuzu_lock and lock_acquired:
                    try:
                        self.close()
                    finally:
                        self.redis_lock.release_lock()

        if cache_config.shared_kuzu_lock:
            async with self._connection_change_lock:
                self.open_connections += 1
                logger.info(f"Open connections after open: {self.open_connections}")
                try:
                    result = blocking_query()
                finally:
                    self.open_connections -= 1
                    logger.info(f"Open connections after close: {self.open_connections}")
                return result
        else:
            result = await loop.run_in_executor(self.executor, blocking_query)
            return result

    def close(self):
        if self.connection:
            del self.connection
            self.connection = None
        if self.db:
            del self.db
            self.db = None
        self._is_closed = True
        logger.info("Ladybug database closed successfully")

    def reopen(self):
        if self._is_closed:
            self._is_closed = False
            self._initialize_connection()
            logger.info("Ladybug database re-opened successfully")

    @asynccontextmanager
    async def get_session(self):
        """
        Get a database session.
        """
        try:
            yield self.connection
        finally:
            pass

    def _parse_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw node result (with JSON properties) into a dictionary."""
        if data.get("properties"):
            try:
                props = json.loads(data["properties"])
                data.pop("properties")
                data.update(props)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse properties JSON for node {data.get('id')}")
        return data

    def _parse_node_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if isinstance(data, dict) and "properties" in data and data["properties"]:
                props = json.loads(data["properties"])
                data.update(props)
                del data["properties"]
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse properties JSON for node {data.get('id')}")
            return data

    def _edge_query_and_params(
        self, from_node: str, to_node: str, relationship_name: str, properties: Dict[str, Any]
    ) -> Tuple[str, dict]:
        """Build the edge creation query and parameters."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        query = """
            MATCH (from:Node), (to:Node)
            WHERE from.id = $from_id AND to.id = $to_id
            MERGE (from)-[r:EDGE {
                relationship_name: $relationship_name
            }]->(to)
            ON CREATE SET
                r.created_at = timestamp($created_at),
                r.updated_at = timestamp($updated_at),
                r.properties = $properties
            ON MATCH SET
                r.updated_at = timestamp($updated_at),
                r.properties = $properties
        """
        params = {
            "from_id": from_node,
            "to_id": to_node,
            "relationship_name": relationship_name,
            "created_at": now,
            "updated_at": now,
            "properties": json.dumps(properties, cls=JSONEncoder),
        }
        return query, params

    async def has_node(self, node_id: str) -> bool:
        query_str = "MATCH (n:Node) WHERE n.id = $id RETURN COUNT(n) > 0"
        result = await self.query(query_str, {"id": node_id})
        return result[0][0] if result else False

    async def add_node(self, node: DataPoint) -> None:
        try:
            properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

            core_properties = {
                "id": str(properties.get("id", "")),
                "name": str(properties.get("name", "")),
                "type": str(properties.get("type", "")),
            }

            for key in core_properties:
                properties.pop(key, None)

            core_properties["properties"] = json.dumps(properties, cls=JSONEncoder)

            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
            fields = []
            params = {}
            for key, value in core_properties.items():
                if value is not None:
                    param_name = f"param_{key}"
                    fields.append(f"{key}: ${param_name}")
                    params[param_name] = value

            fields.extend(
                ["created_at: timestamp($created_at)", "updated_at: timestamp($updated_at)"]
            )
            params.update({"created_at": now, "updated_at": now})

            merge_query = f"""
            MERGE (n:Node {{id: $param_id}})
            ON CREATE SET n += {{{", ".join(fields)}}}
            """
            await self.query(merge_query, params)

        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    async def add_nodes(self, nodes: List[DataPoint]) -> None:
        if not nodes:
            return

        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

            node_params = []
            for node in nodes:
                properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

                core_properties = {
                    "id": str(properties.get("id", "")),
                    "name": str(properties.get("name", "")),
                    "type": str(properties.get("type", "")),
                }

                for key in core_properties:
                    properties.pop(key, None)

                node_params.append(
                    {
                        **core_properties,
                        "properties": json.dumps(properties, cls=JSONEncoder),
                        "created_at": now,
                        "updated_at": now,
                    }
                )

            if node_params:
                merge_query = """
                UNWIND $nodes AS node
                MERGE (n:Node {id: node.id})
                ON CREATE SET
                    n.name = node.name,
                    n.type = node.type,
                    n.properties = node.properties,
                    n.created_at = timestamp(node.created_at),
                    n.updated_at = timestamp(node.updated_at)
                ON MATCH SET
                    n.name = node.name,
                    n.type = node.type,
                    n.properties = node.properties,
                    n.updated_at = timestamp(node.updated_at)
                """
                await self.query(merge_query, {"nodes": node_params})
                logger.debug(f"Processed {len(node_params)} nodes in batch")

        except Exception as e:
            logger.error(f"Failed to add nodes in batch: {e}")
            raise

    async def delete_node(self, node_id: str) -> None:
        query_str = "MATCH (n:Node) WHERE n.id = $id DETACH DELETE n"
        await self.query(query_str, {"id": node_id})

    async def delete_nodes(self, node_ids: List[str]) -> None:
        query_str = "MATCH (n:Node) WHERE n.id IN $ids DETACH DELETE n"
        await self.query(query_str, {"ids": node_ids})

    async def extract_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        query_str = """
        MATCH (n:Node)
        WHERE n.id = $id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            if result and result[0]:
                node_data = self._parse_node(result[0][0])
                return node_data
            return None
        except Exception as e:
            logger.error(f"Failed to extract node {node_id}: {e}")
            return None

    async def extract_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        query_str = """
        MATCH (n:Node)
        WHERE n.id IN $node_ids
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_ids": node_ids})
            nodes = [self._parse_node(row[0]) for row in results if row[0]]
            return nodes
        except Exception as e:
            logger.error(f"Failed to extract nodes: {e}")
            return []

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        query_str = """
        MATCH (from:Node)-[r:EDGE]->(to:Node)
        WHERE from.id = $from_id AND to.id = $to_id AND r.relationship_name = $edge_label
        RETURN COUNT(r) > 0
        """
        result = await self.query(
            query_str, {"from_id": from_node, "to_id": to_node, "edge_label": edge_label}
        )
        return result[0][0] if result else False

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        if not edges:
            return []

        try:
            edge_params = [
                {
                    "from_id": str(from_node),
                    "to_id": str(to_node),
                    "relationship_name": str(edge_label),
                }
                for from_node, to_node, edge_label in edges
            ]

            query = """
            UNWIND $edges AS edge
            MATCH (from:Node)-[r:EDGE]->(to:Node)
            WHERE from.id = edge.from_id
            AND to.id = edge.to_id
            AND r.relationship_name = edge.relationship_name
            RETURN from.id, to.id, r.relationship_name
            """

            results = await self.query(query, {"edges": edge_params})

            existing_edges = [(str(row[0]), str(row[1]), str(row[2])) for row in results]

            logger.debug(f"Found {len(existing_edges)} existing edges out of {len(edges)} checked")
            return existing_edges

        except Exception as e:
            logger.error(f"Failed to check edges in batch: {e}")
            return []

    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = {},
    ) -> None:
        try:
            query, params = self._edge_query_and_params(
                from_node, to_node, relationship_name, edge_properties
            )
            await self.query(query, params)
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            raise

    async def add_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        if not edges:
            return

        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

            edge_params = [
                {
                    "from_id": from_node,
                    "to_id": to_node,
                    "relationship_name": relationship_name,
                    "properties": json.dumps(properties, cls=JSONEncoder),
                    "created_at": now,
                    "updated_at": now,
                }
                for from_node, to_node, relationship_name, properties in edges
            ]

            query = """
            UNWIND $edges AS edge
            MATCH (from:Node), (to:Node)
            WHERE from.id = edge.from_id AND to.id = edge.to_id
            MERGE (from)-[r:EDGE {
                relationship_name: edge.relationship_name
            }]->(to)
            ON CREATE SET
                r.created_at = timestamp(edge.created_at),
                r.updated_at = timestamp(edge.updated_at),
                r.properties = edge.properties
            ON MATCH SET
                r.updated_at = timestamp(edge.updated_at),
                r.properties = edge.properties
            """

            await self.query(query, {"edges": edge_params})

        except Exception as e:
            logger.error(f"Failed to add edges in batch: {e}")
            raise

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        query_str = """
        MATCH (n:Node)-[r]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        },
        r.relationship_name,
        {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    source_node = self._parse_node_properties(row[0])
                    target_node = self._parse_node_properties(row[2])
                    edges.append((source_node, row[1], target_node))
            return edges
        except Exception as e:
            logger.error(f"Failed to get edges for node {node_id}: {e}")
            return []

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        return await self.get_neighbours(node_id)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        query_str = """
        MATCH (n:Node)
        WHERE n.id = $id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            if result and result[0]:
                return self._parse_node(result[0][0])
            return None
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        query_str = """
        MATCH (n:Node)
        WHERE n.id IN $node_ids
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_ids": node_ids})
            return [self._parse_node(row[0]) for row in results if row[0]]
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []

    async def get_neighbours(self, node_id: str) -> List[Dict[str, Any]]:
        query_str = """
        MATCH (n)-[r]-(m)
        WHERE n.id = $id
        RETURN DISTINCT properties(m)
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get neighbours for node {node_id}: {e}")
            return []

    async def get_predecessors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            if edge_label:
                query_str = """
                MATCH (n)<-[r:EDGE]-(m)
                WHERE n.id = $id AND r.relationship_name = $edge_label
                RETURN properties(m)
                """
                params = {"id": str(node_id), "edge_label": edge_label}
            else:
                query_str = """
                MATCH (n)<-[r:EDGE]-(m)
                WHERE n.id = $id
                RETURN properties(m)
                """
                params = {"id": str(node_id)}
            result = await self.query(query_str, params)
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get predecessors for node {node_id}: {e}")
            return []

    async def get_successors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            if edge_label:
                query_str = """
                MATCH (n)-[r:EDGE]->(m)
                WHERE n.id = $id AND r.relationship_name = $edge_label
                RETURN properties(m)
                """
                params = {"id": str(node_id), "edge_label": edge_label}
            else:
                query_str = """
                MATCH (n)-[r:EDGE]->(m)
                WHERE n.id = $id
                RETURN properties(m)
                """
                params = {"id": str(node_id)}
            result = await self.query(query_str, params)
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get successors for node {node_id}: {e}")
            return []

    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        query_str = """
        MATCH (n:Node)-[r:EDGE]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        },
        {
            relationship_name: r.relationship_name,
            properties: r.properties
        },
        {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    processed_rows = []
                    for i, item in enumerate(row):
                        if isinstance(item, dict):
                            if "properties" in item and item["properties"]:
                                try:
                                    props = json.loads(item["properties"])
                                    item.update(props)
                                    del item["properties"]
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON properties for node/edge {i}"
                                    )
                        processed_rows.append(item)
                    edges.append(tuple(processed_rows))
            return edges if edges else []
        except Exception as e:
            logger.error(f"Failed to get connections for node {node_id}: {e}")
            return []

    async def remove_connection_to_predecessors_of(
        self, node_ids: List[str], edge_label: str
    ) -> None:
        query_str = """
        MATCH (n)<-[r:EDGE]-(m)
        WHERE n.id IN $node_ids AND r.relationship_name = $edge_label
        DELETE r
        """
        await self.query(query_str, {"node_ids": node_ids, "edge_label": edge_label})

    async def remove_connection_to_successors_of(
        self, node_ids: List[str], edge_label: str
    ) -> None:
        query_str = """
        MATCH (n)-[r:EDGE]->(m)
        WHERE n.id IN $node_ids AND r.relationship_name = $edge_label
        DELETE r
        """
        await self.query(query_str, {"node_ids": node_ids, "edge_label": edge_label})

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]]:
        import time

        start_time = time.time()

        try:
            nodes_query = """
            MATCH (n:Node)
            RETURN n.id, {
                name: n.name,
                type: n.type,
                properties: n.properties
            }
            """
            nodes = await self.query(nodes_query)
            formatted_nodes = []
            for n in nodes:
                if n[0]:
                    node_id = str(n[0])
                    props = n[1]
                    if props.get("properties"):
                        try:
                            additional_props = json.loads(props["properties"])
                            props.update(additional_props)
                            del props["properties"]
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse properties JSON for node {node_id}")
                    formatted_nodes.append((node_id, props))
            if not formatted_nodes:
                logger.warning("No nodes found in the database")
                return [], []

            edges_query = """
            MATCH (n:Node)-[r]->(m:Node)
            RETURN n.id, m.id, r.relationship_name, r.properties
            """
            edges = await self.query(edges_query)
            formatted_edges = []
            for e in edges:
                if e and len(e) >= 3:
                    source_id = str(e[0])
                    target_id = str(e[1])
                    rel_type = str(e[2])
                    props = {}
                    if len(e) > 3 and e[3]:
                        try:
                            props = json.loads(e[3])
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"Failed to parse edge properties for {source_id}->{target_id}"
                            )
                    formatted_edges.append((source_id, target_id, rel_type, props))

            if formatted_nodes and not formatted_edges:
                logger.debug("No edges found, creating self-referential edges for nodes")
                for node_id, _ in formatted_nodes:
                    formatted_edges.append(
                        (
                            node_id,
                            node_id,
                            "SELF",
                            {
                                "relationship_name": "SELF",
                                "relationship_type": "SELF",
                                "vector_distance": 0.0,
                            },
                        )
                    )

            retrieval_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(nodes)} nodes and {len(edges)} edges in {retrieval_time:.2f} seconds"
            )
            return formatted_nodes, formatted_edges
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            raise

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        label = node_type.__name__
        primary_query = """
            UNWIND $names AS wantedName
            MATCH (n:Node)
            WHERE n.type = $label AND n.name = wantedName
            RETURN DISTINCT n.id
        """
        primary_rows = await self.query(primary_query, {"names": node_name, "label": label})
        primary_ids = [row[0] for row in primary_rows]
        if not primary_ids:
            return [], []

        neighbor_query = """
            MATCH (n:Node)-[:EDGE]-(nbr:Node)
            WHERE n.id IN $ids
            RETURN DISTINCT nbr.id
        """
        nbr_rows = await self.query(neighbor_query, {"ids": primary_ids})
        neighbor_ids = [row[0] for row in nbr_rows]

        all_ids = list({*primary_ids, *neighbor_ids})

        nodes_query = """
            MATCH (n:Node)
            WHERE n.id IN $ids
            RETURN n.id, n.name, n.type, n.properties
        """
        node_rows = await self.query(nodes_query, {"ids": all_ids})
        nodes: List[Tuple[str, dict]] = []
        for node_id, name, typ, props in node_rows:
            data = {"id": node_id, "name": name, "type": typ}
            if props:
                try:
                    data.update(json.loads(props))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON props for node {node_id}")
            nodes.append((node_id, data))

        edges_query = """
            MATCH (a:Node)-[r:EDGE]-(b:Node)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id, b.id, r.relationship_name, r.properties
        """
        edge_rows = await self.query(edges_query, {"ids": all_ids})
        edges: List[Tuple[str, str, str, dict]] = []
        for from_id, to_id, rel_type, props in edge_rows:
            data = {}
            if props:
                try:
                    data = json.loads(props)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON props for edge {from_id}->{to_id}")

            edges.append((from_id, to_id, rel_type, data))

        return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ):
        where_clauses = []
        params = {}

        for i, filter_dict in enumerate(attribute_filters):
            for attr, values in filter_dict.items():
                param_name = f"values_{i}_{attr}"
                where_clauses.append(f"n.{attr} IN ${param_name}")
                params[param_name] = values

        where_clause = " AND ".join(where_clauses)
        nodes_query = f"""
        MATCH (n:Node)
        WHERE {where_clause}
        RETURN n.id, {{
            name: n.name,
            type: n.type,
            properties: n.properties
        }}
        """
        try:
            results = await self.query(nodes_query, params)
            formatted_nodes = []
            for row in results:
                if row and len(row) >= 2:
                    node_id = str(row[0])
                    props = row[1] if isinstance(row[1], dict) else {}
                    if props.get("properties"):
                        try:
                            props.update(json.loads(props["properties"]))
                            del props["properties"]
                        except json.JSONDecodeError:
                            pass
                    formatted_nodes.append((node_id, props))
            return formatted_nodes, []
        except Exception as e:
            logger.error(f"Failed to get filtered graph data: {e}")
            return [], []

    async def get_node_count(self) -> int:
        query_str = "MATCH (n:Node) RETURN COUNT(n)"
        try:
            result = await self.query(query_str)
            return result[0][0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return 0

    async def get_edge_count(self) -> int:
        query_str = "MATCH (n:Node)-[r]->(m:Node) RETURN COUNT(r)"
        try:
            result = await self.query(query_str)
            return result[0][0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get edge count: {e}")
            return 0

    async def delete_all(self) -> None:
        """Delete all nodes and edges from the graph."""
        try:
            self.connection.execute("MATCH (n) DETACH DELETE n")
            logger.info("All nodes and edges deleted from Ladybug database")
        except Exception as e:
            logger.error(f"Failed to delete all data: {e}")
            raise
