"""Shared constants for the migration framework.

Kept dependency-free (only stdlib ``uuid``) so it can be imported from both the
runner and the dataset-listing methods without creating an import cycle.
"""

from uuid import NAMESPACE_DNS, uuid5

# Reserved dataset that anchors the single global ``dataset_database`` row used
# when backend access control is disabled (there are no per-dataset databases in
# that mode). Derived deterministically so it is stable across machines and does
# not collide with the NIL UUID used as a sentinel elsewhere. This dataset is
# hidden from user-facing dataset listings.
GLOBAL_DATASET_ID = uuid5(NAMESPACE_DNS, "global-dataset.cognee.ai")
GLOBAL_DATASET_NAME = "__cognee_global__"
