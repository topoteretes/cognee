from typing import Any, Optional

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment

# Vector collection for the persisted GlobalContextSummary.text index.
GLOBAL_CONTEXT_SUMMARY_COLLECTION = "GlobalContextSummary_text"


async def load_root_text() -> Optional[str]:
    """
    Return the dataset's root GlobalContextSummary text, or None when no root
    exists in scope. Relies on backend access control to scope the query to
    one dataset; within scope a dataset has at most one root.
    """
    fragment = await get_memory_fragment(
        properties_to_project=["id", "text", "type", "is_root"],
        memory_fragment_filter=[{"type": ["GlobalContextSummary"]}],
    )
    for node in fragment.nodes.values():
        if _is_root(node.attributes):
            text = node.attributes.get("text")
            return text or None
    return None


async def search_top_global_context_summaries(
    query: str,
    top_k: int,
    vector_engine: Any,
) -> list[str]:
    """Return up to top_k non-root GlobalContextSummary texts ranked by similarity."""
    if top_k <= 0:
        return []
    try:
        results = await vector_engine.search(
            GLOBAL_CONTEXT_SUMMARY_COLLECTION,
            query,
            limit=top_k + 1,  # +1 in case the root is in the top results
            include_payload=True,
        )
    except CollectionNotFoundError:
        return []

    summaries: list[str] = []
    for result in results or []:
        payload = getattr(result, "payload", None) or {}
        if _is_root(payload):
            continue
        text = payload.get("text")
        if text:
            summaries.append(text)
        if len(summaries) >= top_k:
            break
    return summaries


def format_global_context_prelude(
    root_text: Optional[str],
    top_summaries: list[str],
) -> str:
    """Build the prepend string. Empty when neither input has content."""
    blocks: list[str] = []
    if root_text:
        blocks.append(f"World summary:\n{root_text}")
    if top_summaries:
        joined = "\n\n".join(top_summaries)
        blocks.append(f"Relevant areas:\n{joined}")
    return "\n\n".join(blocks)


def _is_root(attributes: dict) -> bool:
    value = attributes.get("is_root")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
