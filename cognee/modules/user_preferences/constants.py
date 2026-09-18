"""Structural constants for the user-preference store.

Deployment-tunable knobs (alpha, beta, influence) live in ``base_config.py``;
everything here is structural and not meant to be tuned per deployment.
"""

from cognee.modules.improve.constants import USER_PREFERENCES_NODE_SET

# Relationship name of the weighted edges from a preference node to content nodes.
PREFERS_RELATIONSHIP = "prefers"

# NodeSet that groups every preference node, so they can be listed,
# pruned, and deleted together.
PREFERENCE_NODE_SET = USER_PREFERENCES_NODE_SET

# A prefers-edge weight of exactly this value carries no signal.
NEUTRAL_WEIGHT = 0.5

# Distance from neutral below which a prefers edge is pruned on read: an edge
# that no longer says anything goes away, with no half-life to justify.
PREFERENCE_DELETE_THRESHOLD = 0.01

# memify_metadata keys the preference update writes on session QA turns.
# The clock and the evidence are separate: every turn is counted exactly once
# (rated or not), while only turns that carried a usable rating are applied.
PREFERENCE_TURN_COUNTED_KEY = "preference_turn_counted"
PREFERENCE_WEIGHTS_APPLIED_KEY = "preference_weights_applied"

# Cap on the preference node's text; truncation drops the oldest lines.
MAX_PREFERENCE_TEXT_CHARS = 2000

# Prepended at render time, never stored on the node — the node stores data,
# one place owns presentation.
PREFERENCE_RENDER_HEADER = (
    "## What this user prefers\n"
    "Most recent first. When two lines conflict, follow the one nearer the top."
)
