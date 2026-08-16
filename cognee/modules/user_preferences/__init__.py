"""Per-user preference storage inside a dataset's graph.

One ``UserPreference`` node per (user, dataset) plus weighted ``prefers``
edges to content nodes. Written straight through the graph engine, never
embedded, and invisible to retrieval via the ``InternalDataPoint`` marker.
Every import the feature's callers need comes from this package, so it has
one home.
"""

from .constants import (
    MAX_PREFERENCE_TEXT_CHARS,
    NEUTRAL_WEIGHT,
    PREFERENCE_NODE_SET,
    PREFERENCE_RENDER_HEADER,
    PREFERS_RELATIONSHIP,
)
from .models import UserPreference
from .store import (
    delete_prefers_edges,
    load_preference_state,
    preference_node_id,
    upsert_preference_node,
    write_prefers_edges,
)

__all__ = [
    "MAX_PREFERENCE_TEXT_CHARS",
    "NEUTRAL_WEIGHT",
    "PREFERENCE_NODE_SET",
    "PREFERENCE_RENDER_HEADER",
    "PREFERS_RELATIONSHIP",
    "UserPreference",
    "delete_prefers_edges",
    "load_preference_state",
    "preference_node_id",
    "upsert_preference_node",
    "write_prefers_edges",
]
