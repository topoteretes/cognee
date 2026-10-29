import os

_TRUTHY = ("true", "1", "yes")


def search_history_enabled() -> bool:
    """Whether searched questions and their answers are persisted as history.

    Read on every call, not once at import, so a process (or a test) can turn
    history off after cognee is imported. ``COGNEE_LOG_SEARCH_HISTORY`` defaults
    to on; set it to ``false`` for deployments that must not retain query or
    answer text in the relational database.
    """
    return os.getenv("COGNEE_LOG_SEARCH_HISTORY", "true").strip().lower() in _TRUTHY
