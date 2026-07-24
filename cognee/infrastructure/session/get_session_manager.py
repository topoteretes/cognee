from uuid import UUID

from cognee.infrastructure.databases.cache import get_cache_engine
from cognee.infrastructure.session.session_manager import SessionManager


def get_session_manager(dataset_id: str | UUID | None = None) -> SessionManager:
    """
    Return a SessionManager instance.

    Uses the cache engine from get_cache_engine() (Redis or FsCache when
    caching/usage_logging is enabled). If caching is disabled, returns
    a SessionManager with cache_engine=None; all operations will no-op
    and return empty/False as appropriate.

    dataset_id binds the manager to a dataset (falls back to the
    current_dataset_id context variable) so omitted session IDs resolve
    to a per-dataset default session and lifecycle rows carry the dataset.
    """
    cache_engine = get_cache_engine()
    return SessionManager(cache_engine=cache_engine, dataset_id=dataset_id)
