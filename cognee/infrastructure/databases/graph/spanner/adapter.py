import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from datetime import datetime, timezone
from uuid import UUID

from google.cloud import spanner
from google.cloud.spanner_v1.param_types import StructType, StructField, STRING

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder

logger = get_logger()

class SpannerAdapter(GraphDBInterface):
    """
    Adapter for Google Cloud Spanner Graph database operations.
    """

    def __init__(self, project_id: str, instance_id: str, database_id: str, graph_name: str = "CogneeGraph"):
        self.project_id = project_id
        self.instance_id = instance_id
        self.database_id = database_id
        self.graph_name = graph_name
        
        # Initialize Spanner Client
        try:
            self.spanner_client = spanner.Client(project=self.project_id)
            self.instance = self.spanner_client.instance(self.instance_id)
            self.database = self.instance.database(self.database_id)
        except Exception as e:
            logger.error(f"Failed to initialize Spanner Client: {e}")
            raise

    async def initialize(self):
        """
        Async initialization if necessary.
        """
        pass

    async def is_empty(self) -> bool:
        query = f"GRAPH {self.graph_name} MATCH (n) RETURN COUNT(n) AS count"
        results = await self.query(query, {})
        if results and len(results) > 0 and results[0][0] > 0:
            return False
        return True

    async def query(self, query: str, params: Optional[dict] = None) -> List[Any]:
        params = params or {}
        # Convert params to Spanner types where necessary
        param_types = {k: spanner.param_types.STRING for k in params}
        
        rows = []
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(query, params=params, param_types=param_types)
            for row in results:
                rows.append(row)
        return rows

    async def add_node(self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(node, str):
            node_id = node
            props = properties or {}
            name = props.get("name", "")
            node_type = props.get("type", "")
        else:
            props = node.model_dump() if hasattr(node, "model_dump") else vars(node)
            node_id = str(props.get("id", ""))
            name = str(props.get("name", ""))
            node_type = str(props.get("type", ""))

        core_props = ["id", "name", "type"]
        other_props = {k: v for k, v in props.items() if k not in core_props}
        properties_json = json.dumps(other_props, cls=JSONEncoder)

        def insert_node(transaction):
            transaction.insert_or_update(
                table="Node",
                columns=("id", "name", "type", "properties", "created_at", "updated_at"),
                values=[
                    (node_id, name, node_type, properties_json, spanner.COMMIT_TIMESTAMP, spanner.COMMIT_TIMESTAMP)
                ]
            )

        self.database.run_in_transaction(insert_node)

    async def add_nodes(self, nodes: Union[List[Tuple[str, Dict[str, Any]]], List[DataPoint]]) -> None:
        if not nodes:
            return
        
        def insert_nodes(transaction):
            values = []
            for node in nodes:
                if isinstance(node, tuple):
                    node_id, props = node
                    name = props.get("name", "")
                    node_type = props.get("type", "")
                else:
                    props = node.model_dump() if hasattr(node, "model_dump") else vars(node)
                    node_id = str(props.get("id", ""))
                    name = str(props.get("name", ""))
                    node_type = str(props.get("type", ""))

                core_props = ["id", "name", "type"]
                other_props = {k: v for k, v in props.items() if k not in core_props}
                properties_json = json.dumps(other_props, cls=JSONEncoder)
                
                values.append((node_id, name, node_type, properties_json, spanner.COMMIT_TIMESTAMP, spanner.COMMIT_TIMESTAMP))
                
            transaction.insert_or_update(
                table="Node",
                columns=("id", "name", "type", "properties", "created_at", "updated_at"),
                values=values
            )
        
        self.database.run_in_transaction(insert_nodes)

    async def delete_node(self, node_id: str) -> None:
        def delete_txn(transaction):
            query = f"GRAPH {self.graph_name} MATCH (n:Node {{id: @id}}) DETACH DELETE n"
            transaction.execute_update(query, params={"id": node_id}, param_types={"id": spanner.param_types.STRING})
        self.database.run_in_transaction(delete_txn)

    async def delete_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
        def delete_txn(transaction):
            for node_id in node_ids:
                query = f"GRAPH {self.graph_name} MATCH (n:Node {{id: @id}}) DETACH DELETE n"
                transaction.execute_update(query, params={"id": node_id}, param_types={"id": spanner.param_types.STRING})
        self.database.run_in_transaction(delete_txn)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        query = f"GRAPH {self.graph_name} MATCH (n:Node {{id: @id}}) RETURN n.id, n.name, n.type, n.properties"
        rows = await self.query(query, {"id": node_id})
        if rows:
            row = rows[0]
            node_data = {
                "id": row[0],
                "name": row[1],
                "type": row[2]
            }
            if row[3]:
                try:
                    props = json.loads(row[3])
                    node_data.update(props)
                except Exception:
                    pass
            return node_data
        return None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        results = []
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node:
                results.append(node)
        return results

    async def add_edge(self, source_id: str, target_id: str, relationship_name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        props = properties or {}
        properties_json = json.dumps(props, cls=JSONEncoder)
        def insert_edge(transaction):
            transaction.insert_or_update(
                table="Edge",
                columns=("from_id", "to_id", "relationship_name", "properties", "created_at", "updated_at"),
                values=[
                    (source_id, target_id, relationship_name, properties_json, spanner.COMMIT_TIMESTAMP, spanner.COMMIT_TIMESTAMP)
                ]
            )
        self.database.run_in_transaction(insert_edge)

    async def add_edges(self, edges: Union[List[Tuple[str, str, str, Dict[str, Any]]], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]]) -> None:
        if not edges:
            return
        def insert_edges(transaction):
            values = []
            for edge in edges:
                source_id, target_id, relationship_name, props = edge
                properties_json = json.dumps(props or {}, cls=JSONEncoder)
                values.append((source_id, target_id, relationship_name, properties_json, spanner.COMMIT_TIMESTAMP, spanner.COMMIT_TIMESTAMP))
                
            transaction.insert_or_update(
                table="Edge",
                columns=("from_id", "to_id", "relationship_name", "properties", "created_at", "updated_at"),
                values=values
            )
        self.database.run_in_transaction(insert_edges)

    async def delete_graph(self) -> None:
        def delete_all(transaction):
            transaction.execute_update(f"GRAPH {self.graph_name} MATCH (n) DETACH DELETE n")
        self.database.run_in_transaction(delete_all)

    async def get_graph_data(self) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        nodes = []
        edges = []
        
        node_query = f"GRAPH {self.graph_name} MATCH (n:Node) RETURN n.id, n.name, n.type, n.properties"
        node_rows = await self.query(node_query, {})
        for row in node_rows:
            node_data = {"id": row[0], "name": row[1], "type": row[2]}
            if row[3]:
                try:
                    node_data.update(json.loads(row[3]))
                except Exception:
                    pass
            nodes.append((row[0], node_data))
            
        edge_query = f"GRAPH {self.graph_name} MATCH (from:Node)-[r:EDGE]->(to:Node) RETURN from.id, to.id, r.relationship_name, r.properties"
        edge_rows = await self.query(edge_query, {})
        for row in edge_rows:
            props = {}
            if row[3]:
                try:
                    props = json.loads(row[3])
                except Exception:
                    pass
            edges.append((row[0], row[1], row[2], props))
            
        return nodes, edges

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        node_count_query = f"GRAPH {self.graph_name} MATCH (n) RETURN COUNT(n)"
        edge_count_query = f"GRAPH {self.graph_name} MATCH ()-[r]->() RETURN COUNT(r)"
        
        node_count = (await self.query(node_count_query, {}))[0][0]
        edge_count = (await self.query(edge_count_query, {}))[0][0]
        
        return {
            "node_count": node_count,
            "edge_count": edge_count
        }

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        query = f"""
        GRAPH {self.graph_name}
        MATCH (from:Node {{id: @from_id}})-[r:EDGE {{relationship_name: @rel_name}}]->(to:Node {{id: @to_id}})
        RETURN COUNT(r) > 0
        """
        rows = await self.query(query, {"from_id": source_id, "to_id": target_id, "rel_name": relationship_name})
        if rows and rows[0][0]:
            return True
        return False

    async def has_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        existing_edges = []
        for edge in edges:
            source_id, target_id, relationship_name, props = edge
            if await self.has_edge(source_id, target_id, relationship_name):
                existing_edges.append(edge)
        return existing_edges

    async def get_edges(self, node_id: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        query = f"""
        GRAPH {self.graph_name}
        MATCH (from:Node {{id: @id}})-[r:EDGE]->(to:Node)
        RETURN from.id, to.id, r.relationship_name, r.properties
        UNION ALL
        MATCH (from:Node)-[r:EDGE]->(to:Node {{id: @id}})
        RETURN from.id, to.id, r.relationship_name, r.properties
        """
        rows = await self.query(query, {"id": node_id})
        edges = []
        seen = set()
        for row in rows:
            edge_sig = (row[0], row[1], row[2])
            if edge_sig not in seen:
                seen.add(edge_sig)
                props = {}
                if row[3]:
                    try:
                        props = json.loads(row[3])
                    except Exception:
                        pass
                edges.append((row[0], row[1], row[2], props))
        return edges

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        query = f"""
        GRAPH {self.graph_name}
        MATCH (from:Node {{id: @id}})-[:EDGE]-(to:Node)
        RETURN DISTINCT to.id, to.name, to.type, to.properties
        """
        rows = await self.query(query, {"id": node_id})
        neighbors = []
        for row in rows:
            node_data = {
                "id": row[0],
                "name": row[1],
                "type": row[2]
            }
            if row[3]:
                try:
                    node_data.update(json.loads(row[3]))
                except Exception:
                    pass
            neighbors.append(node_data)
        return neighbors

    async def get_nodeset_subgraph(self, node_type: Type[Any], node_name: List[str]) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        raise NotImplementedError("Not implemented for Spanner yet")

    async def get_connections(self, node_id: Union[str, UUID]) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        node_id_str = str(node_id)
        query = f"""
        GRAPH {self.graph_name}
        MATCH (from:Node {{id: @id}})-[r:EDGE]->(to:Node)
        RETURN from.id, from.name, from.type, from.properties,
               to.id, to.name, to.type, to.properties,
               r.relationship_name, r.properties
        """
        rows = await self.query(query, {"id": node_id_str})
        connections = []
        for row in rows:
            from_node = {"id": row[0], "name": row[1], "type": row[2]}
            if row[3]:
                try: from_node.update(json.loads(row[3]))
                except: pass
                
            to_node = {"id": row[4], "name": row[5], "type": row[6]}
            if row[7]:
                try: to_node.update(json.loads(row[7]))
                except: pass
                
            edge_props = {}
            if row[9]:
                try: edge_props = json.loads(row[9])
                except: pass
                
            connections.append((from_node, {"relationship_name": row[8], "properties": edge_props}, to_node))
        return connections

    async def get_filtered_graph_data(self, attribute_filters: List[Dict[str, List[Union[str, int]]]]) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        raise NotImplementedError("Not implemented for Spanner yet")
