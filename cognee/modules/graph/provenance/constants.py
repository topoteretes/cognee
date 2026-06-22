"""Graph-native provenance contract — marker constants and property keys.

Part 0 of the graph-native delete/rollback plan locks the *contract* only:
the names and shapes that Part 1 (storage primitives) and Part 2 (delete /
rollback wiring) will both depend on. Nothing here changes runtime behaviour.

The provenance that the relational ``nodes`` / ``edges`` ledger records today
(which dataset, which data item, and which pipeline run created a graph
artifact) will instead be stamped *onto the graph nodes and edges themselves*
and onto their vector payloads. These constants name the keys that provenance
is stored under and mark artifacts written under the new scheme.
"""

# Bumped whenever the on-graph provenance layout changes in a
# backwards-incompatible way. Part 3 migration uses this to tell graph-native
# artifacts apart from ledger-era ones. Starts at 1.
PROVENANCE_VERSION = 1

# Value stamped on a node/edge to mark it as deletable without the relational
# ledger. Delete/rollback may only take the graph-native path for artifacts
# carrying this mode; everything else falls back to the ledger until migrated.
DELETE_MODE_GRAPH_NATIVE = "graph_native"

# Value representing the legacy behaviour: provenance lives in the relational
# ledger, not on the graph. Used by Part 3 to classify un-migrated artifacts.
DELETE_MODE_LEDGER = "ledger"

# --- Property keys stamped onto graph nodes/edges and vector payloads ---------

# Integer; equals PROVENANCE_VERSION at write time.
PROVENANCE_VERSION_KEY = "provenance_version"

# String; one of DELETE_MODE_GRAPH_NATIVE / DELETE_MODE_LEDGER.
DELETE_MODE_KEY = "delete_mode"

# List[str]; the set of source refs (see refs.make_source_ref) an artifact
# belongs to. A node/edge shared across data items carries multiple refs and
# is hard-deleted only once the last ref is removed.
SOURCE_REFS_KEY = "source_refs"

# List[str]; the set of source-run refs (see refs.make_source_run_ref) that
# touched an artifact. Used by rollback to remove only what a given pipeline
# run introduced.
SOURCE_RUN_REFS_KEY = "source_run_refs"

# List[str]; the dataset ids an artifact belongs to. Source refs are opaque
# hashes of (dataset_id, data_id) and can't be filtered by dataset, so the
# dataset ids are stamped explicitly to support dataset-scoped deletion.
DATASET_IDS_KEY = "dataset_ids"

__all__ = [
    "PROVENANCE_VERSION",
    "DELETE_MODE_GRAPH_NATIVE",
    "DELETE_MODE_LEDGER",
    "PROVENANCE_VERSION_KEY",
    "DELETE_MODE_KEY",
    "SOURCE_REFS_KEY",
    "SOURCE_RUN_REFS_KEY",
    "DATASET_IDS_KEY",
]
