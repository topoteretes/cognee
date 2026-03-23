"""Tool definitions for agent integration.

These are thin, LLM-friendly wrappers around Cognee's V2 API.
They present simple (str -> str) signatures that serializers can
convert to JSON Schema for any framework.
"""

from cognee.api.v2 import remember as v2_remember
from cognee.api.v2 import recall as v2_recall


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
    await v2_remember(data=content, dataset_name=dataset_name)
    return "Remembered."


async def recall(query: str, top_k: int = 5) -> str:
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
        results = await v2_recall(query_text=query, top_k=top_k)
        if results and isinstance(results, list) and len(results) > 0:
            return str(results[0])
        return "No relevant memories found."
    except Exception as e:
        return f"Memory search failed: {e}"


# Registry of all tools — used by serializers and handler
TOOLS = [remember, recall]
