"""
Apache AGE Adapter for Graph Database Operations

This module provides a Python async adapter for Apache AGE (A Graph Extension),
a PostgreSQL extension that adds graph database capabilities with Cypher support.
"""

import json
import asyncio
from typing import Optional, Any, List, Dict, Tuple
from dataclasses import dataclass

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


@dataclass
class NodeData:
    """Represents a graph node."""
    id: str
    properties: Dict[str, Any]


@dataclass
class EdgeData:
    """Represents a graph edge."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]


class ApacheAGEAdapter:
    """
    Async adapter for Apache AGE graph database operations.
    
    Provides methods for:
    - Node CRUD operations
    - Edge CRUD operations
    - Cypher query execution
    - Graph traversal and metrics
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        username: str = "postgres",
        password: str = "password",
        database: str = "cognee",
        graph_name: str = "cognee_graph",
    ):
        """
        Initialize the AGE adapter.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            username: PostgreSQL username
            password: PostgreSQL password
            database: PostgreSQL database name
            graph_name: AGE graph name (schema)
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required. Install with: pip install asyncpg"
            )
            
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.graph_name = graph_name
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self) -> None:
        """
        Create connection pool and initialize AGE.
        
        Automatically creates the graph if it doesn't exist.
        """
        if self.pool is None:
            # Connection initialization callback
            async def init_connection(conn):
                """Initialize each connection in the pool with AGE settings."""
                await conn.execute("LOAD 'age';")
                await conn.execute("SET search_path = ag_catalog, '$user', public;")
            
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                min_size=2,
                max_size=10,
                init=init_connection  # Initialize each connection
            )
            
        # Initialize AGE extension (only once)
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS age;")
        
        # Create graph if it doesn't exist
        await self.create_graph_if_not_exists()
        
        # Create index on id for faster MERGE operations
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    SELECT create_vlabel('{self.graph_name}', 'Node');
                    CREATE INDEX IF NOT EXISTS idx_node_id ON {self.graph_name}.Node(id);
                """)
        except:
            pass
    
    async def create_graph_if_not_exists(self) -> bool:
        """
        Create the graph if it doesn't exist.
        
        Returns:
            True if graph was created, False if it already existed
        """
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(f"SELECT create_graph('{self.graph_name}');")
                print(f"✓ Created graph: {self.graph_name}")
                return True
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"✓ Graph '{self.graph_name}' already exists")
                    return False
                else:
                    raise

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def execute_cypher(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        if self.pool is None:
            await self.connect()
            
        # Wrap Cypher query in AGE SQL syntax
        if params:
            param_str = json.dumps(params)
            wrapped_query = f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    {query}
                $$, '{param_str}') as (result agtype);
            """
        else:
            wrapped_query = f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    {query}
                $$) as (result agtype);
            """
        
        async with self.pool.acquire() as conn:
            # Connections are already initialized with AGE settings in the pool init callback
            rows = await conn.fetch(wrapped_query)
            
            # Parse AGE's agtype results
            results = []
            for row in rows:
                if row['result']:
                    # AGE returns agtype, convert to Python dict
                    result_str = str(row['result'])
                    if result_str and result_str != 'null':
                        try:
                            # Remove AGE-specific type annotations if present
                            # AGE can return things like ::vertex or ::edge
                            if '::' in result_str:
                                result_str = result_str.split('::')[0]
                            
                            result_data = json.loads(result_str)
                            results.append(result_data)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, try to extract just the value
                            results.append({'value': result_str})
                            
            return results

    async def add_node(
        self, 
        node_id: str, 
        labels: List[str], 
        properties: Dict[str, Any]
    ) -> NodeData:
        """
        Add a node to the graph. Uses MERGE to avoid duplicates.
        If a node with the same id already exists, it will be updated with new properties.
        
        Args:
            node_id: Unique node identifier
            labels: Node labels (types)
            properties: Node properties
            
        Returns:
            Created/updated node data
        """
        props = {**properties, 'id': node_id}
        
        # Build property string manually for AGE compatibility
        props_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                props_parts.append(f"{k}: '{v}'")
            elif isinstance(v, bool):
                props_parts.append(f"{k}: {str(v).lower()}")
            elif isinstance(v, (int, float)):
                props_parts.append(f"{k}: {v}")
            elif v is None:
                props_parts.append(f"{k}: null")
            else:
                # For complex types, convert to JSON string
                props_parts.append(f"{k}: '{json.dumps(v)}'")
        
        props_str = ', '.join(props_parts)
        label_str = ':'.join(labels) if labels else 'Node'
        
        # Use MERGE to avoid duplicates - matches on id property
        query = f"""
        MERGE (n:{label_str} {{id: '{node_id}'}})
        SET n = {{{props_str}}}
        RETURN n
        """
        
        results = await self.execute_cypher(query)
        
        return NodeData(
            id=node_id,
            properties=props
        )

    async def add_nodes(self, nodes: List[Tuple[str, List[str], Dict[str, Any]]]) -> None:
        """
        Add multiple nodes in a single batch operation using UNWIND.
        This is significantly faster than calling add_node() multiple times.
        Uses MERGE to avoid duplicates.

        Args:
            nodes: List of tuples (node_id, labels, properties)
        
        Example:
            nodes = [
                ("user_1", ["User"], {"name": "Alice", "age": 30}),
                ("user_2", ["User"], {"name": "Bob", "age": 25}),
            ]
            await adapter.add_nodes(nodes)
        """
        if not nodes:
            return
        
        # Process in batches of 100
        BATCH_SIZE = 100
        
        for i in range(0, len(nodes), BATCH_SIZE):
            batch = nodes[i:i + BATCH_SIZE]
            
            node_data_list = []
            for node_id, labels, properties in batch:
                props = {"id": node_id, **properties}
                props_parts = []
                for k, v in props.items():
                    if isinstance(v, str):
                        props_parts.append(f'{k}: "{v}"')
                    elif isinstance(v, bool):
                        props_parts.append(f'{k}: {str(v).lower()}')
                    elif isinstance(v, (int, float)):
                        props_parts.append(f'{k}: {v}')
                    elif v is None:
                        props_parts.append(f'{k}: null')
                    else:
                        props_parts.append(f'{k}: "{json.dumps(v)}"')
                props_str = '{' + ', '.join(props_parts) + '}'
                label_str = ':'.join(labels) if labels else "Node"
                node_data_list.append(f'{{id: "{node_id}", props: {props_str}, label: "{label_str}"}}')
            
            unwind_data = '[' + ', '.join(node_data_list) + ']'
            
            all_prop_keys = set()
            for node_id, labels, properties in batch:
                all_prop_keys.update(properties.keys())
            all_prop_keys.add('id')
            
            set_clauses = [f"n.{key} = node_data.props.{key}" for key in sorted(all_prop_keys)]
            set_clause = "SET " + ", ".join(set_clauses)
            
            common_label = batch[0][1][0] if batch[0][1] else "Node"
            query = f"""
            UNWIND {unwind_data} AS node_data
            MERGE (n {{id: node_data.id}})
            {set_clause}
            """
            await self.execute_cypher(query)

    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node data or None if not found
        """
        query = f"MATCH (n {{id: '{node_id}'}}) RETURN n"
        results = await self.execute_cypher(query)
        
        if results:
            node_data = results[0]
            # Extract properties from AGE vertex structure
            if isinstance(node_data, dict) and 'properties' in node_data:
                props = node_data['properties']
            else:
                props = node_data
            
            return NodeData(
                id=props.get('id', node_id),
                properties=props
            )
        return None

    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if deleted, False if not found
        """
        query = f"MATCH (n {{id: '{node_id}'}}) DETACH DELETE n"
        await self.execute_cypher(query)
        return True

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> EdgeData:
        """
        Add an edge between two nodes. Uses MERGE to avoid duplicates.
        If an edge already exists between the same nodes with the same type, 
        it will be updated with new properties.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Relationship type/name
            properties: Edge properties
            
        Returns:
            Created/updated edge data
        """
        props = properties or {}
        
        # Build property string manually for AGE compatibility
        if props:
            props_parts = []
            for k, v in props.items():
                if isinstance(v, str):
                    props_parts.append(f"{k}: '{v}'")
                elif isinstance(v, bool):
                    props_parts.append(f"{k}: {str(v).lower()}")
                elif isinstance(v, (int, float)):
                    props_parts.append(f"{k}: {v}")
                elif v is None:
                    props_parts.append(f"{k}: null")
                else:
                    # For complex types, convert to JSON string
                    props_parts.append(f"{k}: '{json.dumps(v)}'")
            props_str = ', '.join(props_parts)
            
            # Use MERGE to avoid duplicate edges
            query = f"""
            MATCH (a {{id: '{source_id}'}}), (b {{id: '{target_id}'}})
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r = {{{props_str}}}
            RETURN r
            """
        else:
            # Use MERGE without properties
            query = f"""
            MATCH (a {{id: '{source_id}'}}), (b {{id: '{target_id}'}})
            MERGE (a)-[r:{relationship_type}]->(b)
            RETURN r
            """
        
        await self.execute_cypher(query)
        
        return EdgeData(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=props
        )

    async def add_edges(self, edges: List[Tuple[str, str, str, Optional[Dict[str, Any]]]]) -> None:
        """
        Add multiple edges in a single batch operation using UNWIND.
        This is significantly faster than calling add_edge() multiple times.
        Uses MERGE to avoid duplicates.

        Args:
            edges: List of tuples (source_id, target_id, relationship_type, properties)
        
        Example:
            edges = [
                ("user_1", "user_2", "KNOWS", {"since": 2020}),
                ("user_2", "user_3", "FOLLOWS", {"weight": 0.5}),
            ]
            await adapter.add_edges(edges)
        """
        if not edges:
            return
        
        # Group edges by relationship type for efficiency
        edges_by_type = {}
        for source_id, target_id, rel_type, properties in edges:
            if rel_type not in edges_by_type:
                edges_by_type[rel_type] = []
            edges_by_type[rel_type].append({
                "source_id": source_id,
                "target_id": target_id,
                "properties": properties or {}
            })
        
        # Process each relationship type in batches
        BATCH_SIZE = 100  # Smaller batches to avoid huge query strings
        
        for rel_type, edge_list in edges_by_type.items():
            # Get all unique property keys for this relationship type
            all_prop_keys = set()
            for edge in edge_list:
                all_prop_keys.update(edge["properties"].keys())
            
            # Process in batches
            for i in range(0, len(edge_list), BATCH_SIZE):
                batch = edge_list[i:i + BATCH_SIZE]
                
                # Build VALUES clause for batch MERGE
                values_parts = []
                for edge in batch:
                    props = edge["properties"]
                    # Build property map for this edge
                    props_cypher_parts = []
                    for key in all_prop_keys:
                        value = props.get(key)
                        if value is None:
                            props_cypher_parts.append(f'{key}: null')
                        elif isinstance(value, str):
                            # Escape quotes in strings
                            escaped = value.replace('"', '\\"')
                            props_cypher_parts.append(f'{key}: "{escaped}"')
                        elif isinstance(value, bool):
                            props_cypher_parts.append(f'{key}: {str(value).lower()}')
                        elif isinstance(value, (int, float)):
                            props_cypher_parts.append(f'{key}: {value}')
                    props_str = ', '.join(props_cypher_parts)
                    
                    values_parts.append(f'{{src: "{edge["source_id"]}", tgt: "{edge["target_id"]}", props: {{{props_str}}}}}')
                
                # Build UNWIND query (AGE requires inline data, not parameters)
                values_list = '[' + ', '.join(values_parts) + ']'
                
                # Build SET clause with explicit assignments (AGE doesn't support SET r = map)
                if all_prop_keys:
                    set_parts = [f'r.{key} = edge.props.{key}' for key in all_prop_keys]
                    set_clause = 'SET ' + ', '.join(set_parts)
                else:
                    set_clause = ''
                
                query = f"""
                UNWIND {values_list} AS edge
                MATCH (a {{id: edge.src}}), (b {{id: edge.tgt}})
                MERGE (a)-[r:{rel_type}]->(b)
                {set_clause}
                """
                
                await self.execute_cypher(query)

    async def get_edges(self, node_id: str) -> List[EdgeData]:
        """
        Get all edges connected to a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of edge data
        """
        query = f"""
        MATCH (a {{id: '{node_id}'}})-[r]-(b)
        RETURN {{source: a.id, target: b.id, rel_type: type(r), props: properties(r)}}
        """
        
        results = await self.execute_cypher(query)
        
        edges = []
        for result in results:
            # Result is directly the edge data map
            edges.append(EdgeData(
                source_id=result.get('source', ''),
                target_id=result.get('target', ''),
                relationship_type=result.get('rel_type', ''),
                properties=result.get('props', {})
            ))
        
        return edges

    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        """
        Get all neighboring nodes.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of neighbor nodes
        """
        # Use simple map return instead of full vertex object for better performance
        query = f"""
        MATCH (n {{id: '{node_id}'}})-[]-(neighbor)
        RETURN DISTINCT {{id: neighbor.id, properties: properties(neighbor)}}
        """
        
        results = await self.execute_cypher(query)
        
        neighbors = []
        for result in results:
            # Result is already a simple map
            neighbors.append(NodeData(
                id=result.get('id', ''),
                properties=result.get('properties', {})
            ))
        
        return neighbors

    async def count_nodes(self) -> int:
        """
        Count total nodes in the graph.
        
        Returns:
            Number of nodes
        """
        query = "MATCH (n) RETURN {count: count(n)}"
        results = await self.execute_cypher(query)
        
        if results:
            return results[0].get('count', 0)
        return 0

    async def count_edges(self) -> int:
        """
        Count total edges in the graph.
        
        Returns:
            Number of edges
        """
        query = "MATCH ()-[r]->() RETURN {count: count(r)}"
        results = await self.execute_cypher(query)
        
        if results:
            return results[0].get('count', 0)
        return 0

    async def clear_graph(self) -> None:
        """
        Delete all nodes and edges from the graph.
        
        Note: This only removes the data. The graph schema and tables remain.
        Use drop_graph() to completely remove the graph including its tables.
        """
        query = "MATCH (n) DETACH DELETE n"
        await self.execute_cypher(query)

    async def drop_graph(self, recreate: bool = False) -> None:
        """
        Completely drop the graph including its schema and tables from PostgreSQL.
        
        Args:
            recreate: If True, recreates an empty graph immediately after dropping
        
        Warning: This permanently removes all graph data, schema, and tables.
        """
        async with self.pool.acquire() as conn:
            await conn.execute("SET search_path = ag_catalog, '$user', public;")
            await conn.execute("LOAD 'age';")
            
            try:
                await conn.execute(f"SELECT drop_graph('{self.graph_name}', true);")
            except asyncpg.exceptions.UndefinedObjectError:
                # Graph doesn't exist, nothing to drop
                pass
            except Exception as e:
                raise Exception(f"Error dropping graph '{self.graph_name}': {e}") from e
        
        # Recreate if requested
        if recreate:
            await self.create_graph_if_not_exists()
    
    async def list_all_graphs(self) -> List[str]:
        """
        List all Apache AGE graphs in the database.
        
        Returns:
            List of graph names
        """
        async with self.pool.acquire() as conn:
            await conn.execute("SET search_path = ag_catalog, '$user', public;")
            await conn.execute("LOAD 'age';")
            
            # Query ag_catalog.ag_graph to get all graphs
            result = await conn.fetch(
                "SELECT name FROM ag_catalog.ag_graph ORDER BY name;"
            )
            
            return [row['name'] for row in result]
    
    async def drop_all_graphs(self) -> List[str]:
        """
        Drop ALL Apache AGE graphs in the database.
        
        Returns:
            List of dropped graph names
        
        Warning: This permanently removes ALL graphs from the database!
        """
        graphs = await self.list_all_graphs()
        
        async with self.pool.acquire() as conn:
            await conn.execute("SET search_path = ag_catalog, '$user', public;")
            await conn.execute("LOAD 'age';")
            
            dropped = []
            for graph_name in graphs:
                try:
                    await conn.execute(f"SELECT drop_graph('{graph_name}', true);")
                    dropped.append(graph_name)
                    print(f"✓ Dropped graph: {graph_name}")
                except Exception as e:
                    print(f"✗ Failed to drop graph '{graph_name}': {e}")
            
            return dropped

    async def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node and edge counts
        """
        num_nodes = await self.count_nodes()
        num_edges = await self.count_edges()
        
        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'mean_degree': (2 * num_edges / num_nodes) if num_nodes > 0 else 0
        }


