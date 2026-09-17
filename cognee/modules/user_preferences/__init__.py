"""Per-user preference storage inside a dataset's graph.

One ``UserPreference`` node per (user, dataset) plus weighted ``prefers``
edges to content nodes. Written straight through the graph engine, never
embedded, and invisible to retrieval via the ``InternalDataPoint`` marker.
The write path (``update.py``) is deliberately not re-exported here: it pulls
in ``tasks.memify``, which imports graph modules that import this package —
callers reach it directly via ``cognee.modules.user_preferences.update``.
"""

from .lookup import (
    load_active_preference_lines,
    load_preference_text,
    load_preference_weights,
    warm_preference_cache,
)
from .models import UserPreference
from .store import preference_node_id
from .weights import personal_factor

__all__ = [
    "UserPreference",
    "load_active_preference_lines",
    "load_preference_text",
    "load_preference_weights",
    "personal_factor",
    "preference_node_id",
    "warm_preference_cache",
]
