"""Retriever for CODE_GRAPH search type.

Looks up code symbols (functions, classes, files) by name using the vector
index, then returns their first-degree graph neighbours as context.
"""

from typing import Any, List, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.shared.logging_utils import get_logger

logger = get_logger("CodeGraphRetriever")

_CODE_COLLECTIONS = ("CodeFunction_name", "CodeClass_name", "CodeFile_file_path")


class CodeGraphRetriever(BaseRetriever):
    """Return graph neighbours of the best-matching code symbol."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    async def get_retrieved_objects(self, query: str) -> List[Any]:
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()

        hits: list = []
        for collection in _CODE_COLLECTIONS:
            try:
                results = await vector_engine.search(collection, query, limit=self.top_k)
                hits.extend(results)
            except Exception:
                pass

        if not hits:
            return []

        # Deduplicate by id and gather graph neighbours
        seen: set = set()
        neighbours: list = []
        for hit in hits:
            node_id = getattr(hit, "id", None) or (hit.get("id") if isinstance(hit, dict) else None)
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            try:
                node_neighbours = await graph_engine.get_neighbours(node_id)
                neighbours.extend(node_neighbours or [])
            except Exception:
                pass

        return hits + neighbours

    async def get_context_from_objects(self, query: str, retrieved_objects: List[Any]) -> str:
        if not retrieved_objects:
            return ""
        lines = []
        for obj in retrieved_objects:
            if isinstance(obj, dict):
                lines.append(str(obj))
            else:
                name = getattr(obj, "name", None) or getattr(obj, "file_path", None) or str(obj)
                node_type = getattr(obj, "type", "node")
                lines.append(f"[{node_type}] {name}")
        return "\n".join(lines)

    async def get_completion_from_context(
        self, query: str, retrieved_objects: List[Any], context: Optional[str]
    ) -> Optional[str]:
        return context
