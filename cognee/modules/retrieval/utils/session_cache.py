from typing import Optional, List, Dict, Any
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.exceptions import CacheConnectionError
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_cache")


async def save_conversation_history(
    query: str,
    context_summary: str,
    answer: str,
    session_id: Optional[str] = None,
) -> bool:
    """
    Saves Q&A interaction to the session cache if user is authenticated and caching is enabled.

    Handles cache unavailability gracefully by logging warnings instead of failing.

    Parameters:
    -----------

        - query (str): The user's query/question.
        - context_summary (str): Summarized context used for generating the answer.
        - answer (str): The generated answer/completion.
        - session_id (Optional[str]): Session identifier. Defaults to 'default_session' if None.

    Returns:
    --------

        - bool: True if successfully saved to cache, False otherwise.
    """
    try:
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)

        if not (user_id and cache_config.caching):
            logger.debug("Session caching disabled or user not authenticated")
            return False

        if session_id is None:
            session_id = "default_session"

        from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

        cache_engine = get_cache_engine()

        if cache_engine is None:
            logger.warning("Cache engine not available, skipping session save")
            return False

        await cache_engine.add_qa(
            str(user_id),
            session_id=session_id,
            question=query,
            context=context_summary,
            answer=answer,
        )

        logger.info(
            f"Successfully saved Q&A to session cache: user_id={user_id}, session_id={session_id}"
        )
        return True

    except CacheConnectionError as e:
        logger.warning(f"Cache unavailable, continuing without session save: {e.message}")
        return False

    except Exception as e:
        logger.error(
            f"Unexpected error saving to session cache: {type(e).__name__}: {str(e)}. Continuing without caching."
        )
        return False


async def get_conversation_history(
    session_id: Optional[str] = None,
) -> str:
    """
    Retrieves conversation history from cache and formats it as text.

    Returns formatted conversation history with time, question, context, and answer
    for the last N Q&A pairs (N is determined by cache engine default).

    Parameters:
    -----------

        - session_id (Optional[str]): Session identifier. Defaults to 'default_session' if None.

    Returns:
    --------

        - str: Formatted conversation history string, or empty string if no history or error.

    Format:
    -------

        Previous conversation:

        [2024-01-15 10:30:45]
        QUESTION: What is X?
        CONTEXT: X is a concept...
        ANSWER: X is...

        [2024-01-15 10:31:20]
        QUESTION: How does Y work?
        CONTEXT: Y is related to...
        ANSWER: Y works by...
    """
    try:
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)

        if not (user_id and cache_config.caching):
            logger.debug("Session caching disabled or user not authenticated")
            return ""

        if session_id is None:
            session_id = "default_session"

        from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

        cache_engine = get_cache_engine()

        if cache_engine is None:
            logger.warning("Cache engine not available, skipping conversation history retrieval")
            return ""

        history_entries = await cache_engine.get_latest_qa(str(user_id), session_id)

        if not history_entries:
            logger.debug("No conversation history found")
            return ""

        history_text = "Previous conversation:\n\n"
        for entry in history_entries:
            history_text += f"[{entry.get('time', 'Unknown time')}]\n"
            history_text += f"QUESTION: {entry.get('question', '')}\n"
            history_text += f"CONTEXT: {entry.get('context', '')}\n"
            history_text += f"ANSWER: {entry.get('answer', '')}\n\n"

        logger.debug(f"Retrieved {len(history_entries)} conversation history entries")
        return history_text

    except CacheConnectionError as e:
        logger.warning(f"Cache unavailable, continuing without conversation history: {e.message}")
        return ""

    except Exception as e:
        logger.warning(
            f"Unexpected error retrieving conversation history: {type(e).__name__}: {str(e)}"
        )
        return ""
