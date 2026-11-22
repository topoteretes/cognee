"""Deployment gates shared by search selection, routing and execution."""

import os

from cognee.modules.search.types import SearchType

CYPHER_SEARCH_TYPES = (SearchType.CYPHER, SearchType.NATURAL_LANGUAGE)


def cypher_queries_allowed() -> bool:
    """Raw Cypher (CYPHER, NATURAL_LANGUAGE) is opt-in through ALLOW_CYPHER_QUERY=true."""
    return os.getenv("ALLOW_CYPHER_QUERY", "false").strip().lower() in ("true", "1", "yes")
