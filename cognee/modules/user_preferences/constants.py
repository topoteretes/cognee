"""Structural constants for the user-preference store.

Deployment-tunable knobs (alpha, beta, influence) live in ``base_config.py``;
everything here is structural and not meant to be tuned per deployment.
"""

# Relationship name of the weighted edges from a preference node to content nodes.
PREFERS_RELATIONSHIP = "prefers"

# NodeSet that groups every preference node, so they can be listed,
# pruned, and deleted together.
PREFERENCE_NODE_SET = "user_preferences"

# A prefers-edge weight of exactly this value carries no signal.
NEUTRAL_WEIGHT = 0.5

# Cap on the preference node's text; truncation drops the oldest lines.
MAX_PREFERENCE_TEXT_CHARS = 2000

# Prepended at render time, never stored on the node — the node stores data,
# one place owns presentation.
PREFERENCE_RENDER_HEADER = (
    "## What this user prefers\n"
    "Most recent first. When two lines conflict, follow the one nearer the top."
)
