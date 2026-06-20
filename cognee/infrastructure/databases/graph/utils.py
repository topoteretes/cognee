"""Utility functions for graph database operations."""

from typing import Any, List, Dict


def normalize_graph_result(result: List[Any], columns: List[str]) -> List[Dict[str, Any]]:
    """
    Normalize graph query results to a consistent dict format.

    Ladybug/Kuzu returns List[Tuple], while Neo4j and Neptune return List[Dict[str, Any]].
    """
    if not result:
        return []
    if isinstance(result[0], tuple):
        return [dict(zip(columns, row)) for row in result]
    return result
