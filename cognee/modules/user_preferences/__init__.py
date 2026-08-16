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
    PREFERENCE_DELETE_THRESHOLD,
    PREFERENCE_NODE_SET,
    PREFERENCE_OVERFETCH,
    PREFERENCE_RENDER_HEADER,
    PREFERENCE_TURN_COUNTED_KEY,
    PREFERENCE_WEIGHTS_APPLIED_KEY,
    PREFERS_RELATIONSHIP,
)
from .lookup import load_active_preferences
from .models import UserPreference
from .store import (
    delete_prefers_edges,
    load_preference_state,
    preference_node_id,
    upsert_preference_node,
    write_prefers_edges,
)
from .update import PreferenceUpdateResult, update_user_preferences
from .weights import effective_weight, personal_factor

__all__ = [
    "MAX_PREFERENCE_TEXT_CHARS",
    "NEUTRAL_WEIGHT",
    "PREFERENCE_DELETE_THRESHOLD",
    "PREFERENCE_NODE_SET",
    "PREFERENCE_OVERFETCH",
    "PREFERENCE_RENDER_HEADER",
    "PREFERENCE_TURN_COUNTED_KEY",
    "PREFERENCE_WEIGHTS_APPLIED_KEY",
    "PREFERS_RELATIONSHIP",
    "PreferenceUpdateResult",
    "UserPreference",
    "delete_prefers_edges",
    "effective_weight",
    "load_active_preferences",
    "load_preference_state",
    "personal_factor",
    "preference_node_id",
    "update_user_preferences",
    "upsert_preference_node",
    "write_prefers_edges",
]
