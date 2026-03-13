"""Utility functions for graph database operations."""

from typing import Any, List, Dict


def normalize_graph_result(result: List[Any], columns: List[str]) -> List[Dict[str, Any]]:
    """
    Normalize graph query results to a consistent dict format.

    Kuzu returns List[Tuple], while Neo4j and Neptune return List[Dict[str, Any]].
    This function converts tuple results to dicts for consistent handling.

    Parameters
    ----------
    result : List[Any]
        Query result from graph database (list of tuples or dicts)
    columns : List[str]
        Column names corresponding to tuple positions

    Returns
    -------
    List[Dict[str, Any]]
        Normalized result as list of dictionaries

    Examples
    --------
    >>> # Kuzu tuple result
    >>> result = [("chunk-1", "doc-1", "doc.pdf", "pdf")]
    >>> normalize_graph_result(result, ["chunk_id", "doc_id", "doc_name", "doc_type"])
    [{"chunk_id": "chunk-1", "doc_id": "doc-1", "doc_name": "doc.pdf", "doc_type": "pdf"}]

    >>> # Neo4j dict result (already normalized)
    >>> result = [{"chunk_id": "chunk-1", "doc_id": "doc-1"}]
    >>> normalize_graph_result(result, ["chunk_id", "doc_id"])
    [{"chunk_id": "chunk-1", "doc_id": "doc-1"}]
    """
    if result and isinstance(result[0], tuple):
        return [dict(zip(columns, row)) for row in result]
    return result
