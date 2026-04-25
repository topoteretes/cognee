"""Callable memory search: lets the agent run a fresh triplet lookup mid-loop."""

from typing import Any, Dict

from cognee.modules.engine.models import Tool
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
)
from cognee.modules.tools.errors import ToolInvocationError
from cognee.modules.tools.registry import register_builtin_tool


TOOL = Tool(
    name="memory_search",
    description=(
        "Run a fresh semantic search over the knowledge graph and return the "
        "relevant triplets as text. Use when the initial context is insufficient "
        "or you need to pivot to a different topic."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {
                "type": "integer",
                "description": "Number of triplets to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler_ref="cognee.modules.tools.builtin.memory_search.handler",
    permission_required="read",
    readonly_hint=True,
)


async def handler(args: Dict[str, Any], **_) -> str:
    query = args.get("query")
    if not query:
        raise ToolInvocationError("memory_search requires a 'query' argument")
    raw_top_k = args.get("top_k", 5)
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError) as exc:
        raise ToolInvocationError("memory_search 'top_k' must be an integer") from exc
    if top_k <= 0:
        raise ToolInvocationError("memory_search 'top_k' must be greater than 0")

    triplets = await brute_force_triplet_search(query, top_k=top_k)
    if not triplets:
        return "(no results)"
    return await resolve_edges_to_text(triplets)


register_builtin_tool(TOOL)
