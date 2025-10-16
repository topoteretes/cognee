from typing import Optional
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig


async def save_to_session_cache(
    query: str,
    context_summary: str,
    answer: str,
    session_id: Optional[str] = None,
) -> None:
    """
    Saves Q&A interaction to the session cache if user is authenticated and caching is enabled.

    Parameters:
    -----------

        - query (str): The user's query/question.
        - context_summary (str): Summarized context used for generating the answer.
        - answer (str): The generated answer/completion.
        - session_id (Optional[str]): Session identifier. Defaults to 'default_session' if None.

    Returns:
    --------

        - None: This function performs a side effect (saving to cache) and returns nothing.
    """
    cache_config = CacheConfig()
    user = session_user.get()
    user_id = getattr(user, "id", None)

    if not (user_id and cache_config.caching):
        return

    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

    cache_engine = get_cache_engine()
    if session_id is None:
        session_id = "default_session"

    await cache_engine.add_qa(
        str(user_id),
        session_id=session_id,
        question=query,
        context=context_summary,
        answer=answer,
    )

