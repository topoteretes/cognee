"""Graph-native provenance contract (Part 0).

Shared types, helpers, and marker constants that the graph-native
delete/rollback work depends on. Part 0 ships the contract only — importing
this package has no runtime side effects and changes no behaviour.

See ``temp/delete_improvements/part0_contract.md`` for the full design and
``temp/delete_improvements/phases_overview.md`` for the four-part plan.
"""

from .constants import (
    DATASET_IDS_KEY,
    DELETE_MODE_GRAPH_NATIVE,
    DELETE_MODE_KEY,
    DELETE_MODE_LEDGER,
    PROVENANCE_VERSION,
    PROVENANCE_VERSION_KEY,
    SOURCE_REFS_KEY,
    SOURCE_RUN_REFS_KEY,
)
from .exceptions import UnsupportedProvenanceCapability
from .refs import make_source_ref, make_source_run_ref
from .results import ProvenanceDeleteResult
from .snapshots import EdgeDeleteData, EdgeIdentity, NodeDeleteData

__all__ = [
    # constants
    "PROVENANCE_VERSION",
    "DELETE_MODE_GRAPH_NATIVE",
    "DELETE_MODE_LEDGER",
    "PROVENANCE_VERSION_KEY",
    "DELETE_MODE_KEY",
    "SOURCE_REFS_KEY",
    "SOURCE_RUN_REFS_KEY",
    "DATASET_IDS_KEY",
    # exceptions
    "UnsupportedProvenanceCapability",
    # refs
    "make_source_ref",
    "make_source_run_ref",
    # snapshots
    "EdgeIdentity",
    "NodeDeleteData",
    "EdgeDeleteData",
    # results
    "ProvenanceDeleteResult",
]
