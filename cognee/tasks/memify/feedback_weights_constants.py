"""Keys the feedback-weight stage writes into ``SessionQAEntry.memify_metadata``.

``feedback_weights_applied`` is the row's done marker (the only key other code reads).
The remaining keys are this stage's own bookkeeping so a partially applied row moves each
graph element exactly once across retries instead of compounding.
"""

# Done marker: True once every surviving element has moved (or the attempt cap was reached).
MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY = "feedback_weights_applied"

# Element ids whose weight already moved for this row (lists of str).
MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_NODE_IDS_KEY = "feedback_weights_applied_node_ids"
MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_EDGE_IDS_KEY = "feedback_weights_applied_edge_ids"

# The 1-5 rating those ids were moved with. A different rating on the same row (re-rated
# feedback) starts the row over; the same rating is a no-op.
MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_SCORE_KEY = "feedback_weights_applied_score"

# Number of runs that tried to write this row's weights (int).
MEMIFY_METADATA_FEEDBACK_WEIGHTS_ATTEMPTS_KEY = "feedback_weights_attempts"

# Ids the last run could not find in its graph (list of str, observability only —
# retries recompute pending from the applied sets). "Not found" conflates deleted
# with living in another dataset's graph, so these ids keep the row pending until
# the attempt cap: an improve on the dataset that HAS them can consume the row.
MEMIFY_METADATA_FEEDBACK_WEIGHTS_PRUNED_IDS_KEY = "feedback_weights_pruned_ids"

# After this many attempts a row is marked done even if some writes never succeeded
# or some ids were never found.
FEEDBACK_WEIGHTS_MAX_ATTEMPTS = 3

# An implicit rating (inferred from the user's next turn) moves weights at half the
# learning rate of an explicit one.
IMPLICIT_FEEDBACK_ALPHA_FACTOR = 0.5

FEEDBACK_SOURCE_EXPLICIT = "explicit"
FEEDBACK_SOURCE_IMPLICIT = "implicit"
