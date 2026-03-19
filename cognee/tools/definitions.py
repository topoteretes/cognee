"""Canonical tool definitions for Cognee memory.

These are plain async functions that wrap existing Cognee APIs.
They are used directly (Tier 2) and by serializers (Tier 3).
"""

import asyncio
from typing import Optional

import cognee
from cognee.modules.search.types import SearchType


async def remember(content: str, dataset_name: str = "main_dataset") -> str:
    """Save information to memory for future retrieval.

    Use this when the user shares preferences, decisions, facts, or anything
    worth recalling later. The information is ingested and processed into
    Cognee's knowledge graph.

    Parameters
    ----------
    content : str
        The text to remember. Can be a fact, a conversation excerpt,
        a user preference, a decision, or any free-form text.
    dataset_name : str, optional
        Dataset to store the memory in. Defaults to "main_dataset".

    Returns
    -------
    str
        Confirmation message.
    """
    await cognee.add(content, dataset_name=dataset_name)
    asyncio.create_task(_background_cognify())
    return "Remembered."


async def search_memory(query: str, top_k: int = 5) -> str:
    """Search memory for relevant information.

    Use this when the user asks a question that might be answered by
    previously stored information, or when context from past conversations
    would be helpful.

    Parameters
    ----------
    query : str
        The search query in natural language.
    top_k : int, optional
        Maximum number of results to return. Defaults to 5.

    Returns
    -------
    str
        Search results as a string the agent can use directly.
    """
    try:
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.GRAPH_COMPLETION,
            top_k=top_k,
        )
        if results and isinstance(results, list) and len(results) > 0:
            return str(results[0])
        return "No relevant memories found."
    except Exception as e:
        return f"Memory search failed: {e}"


async def _background_cognify():
    """Run cognify in the background. Errors are logged, not raised."""
    from cognee.shared.logging_utils import get_logger

    logger = get_logger()
    try:
        await cognee.cognify()
    except Exception as e:
        logger.warning(f"Background cognify failed: {e}")


# Registry of all tools — used by serializers and handler
TOOLS = [remember, search_memory]
