from cognee.infrastructure.databases.cache import get_cache_engine
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.infrastructure.session.session_scope import INHERIT_DATASET, DatasetScopeArg


def get_session_manager(dataset_id: DatasetScopeArg = INHERIT_DATASET) -> SessionManager:
    """
    Return a SessionManager instance.

    Uses the cache engine from get_cache_engine() (Redis or FsCache when
    caching/usage_logging is enabled). If caching is disabled, returns
    a SessionManager with cache_engine=None; all operations will no-op
    and return empty/False as appropriate.
    """
    cache_engine = get_cache_engine()
    return SessionManager(cache_engine=cache_engine, dataset_id=dataset_id)
