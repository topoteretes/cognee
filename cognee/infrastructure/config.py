from typing import Optional
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
import os

_graph_db = None


def get_graph_db():
    """Get the configured graph database adapter."""
    global _graph_db
    if _graph_db is None:
        # Read from environment variables or config file
        db_type = os.getenv("GRAPH_DATABASE_PROVIDER", "neo4j")

        if db_type == "neo4j":
            _graph_db = Neo4jAdapter(
                graph_database_url=os.getenv("GRAPH_DATABASE_URL"),
                graph_database_username=os.getenv("GRAPH_DATABASE_USERNAME"),
                graph_database_password=os.getenv("GRAPH_DATABASE_PASSWORD"),
            )
        elif db_type == "kuzu":
            _graph_db = KuzuAdapter(db_path=os.getenv("KUZU_DB_PATH"))
        elif db_type == "networkx":
            _graph_db = NetworkXAdapter()
        else:
            raise ValueError(f"Unsupported graph database type: {db_type}")

    return _graph_db
