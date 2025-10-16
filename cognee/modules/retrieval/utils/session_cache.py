from typing import Optional
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.exceptions import CacheConnectionError
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_cache")


async def save_to_session_cache(
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
        
        logger.info(f"Successfully saved Q&A to session cache: user_id={user_id}, session_id={session_id}")
        return True

    except CacheConnectionError as e:
        logger.warning(f"Cache unavailable, continuing without session save: {e.message}")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error saving to session cache: {type(e).__name__}: {str(e)}. Continuing without caching.")
        return False
