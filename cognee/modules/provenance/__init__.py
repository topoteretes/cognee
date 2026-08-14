"""Audit-grade provenance ledger (append-only, tamper-evident).

This module is the fourth — and only audit-grade — provenance system in cognee.
It is deliberately distinct from the other three:

- ``cognee/infrastructure/databases/provenance/`` — graph source-refs used for
  delete/rollback ownership (``source_ref:v1:{dataset_id}:{data_id}`` keys).
- ``source_*`` DataPoint stamping — write-time attribution on graph nodes.
- the memory-provenance projection — actor-centric views over session memory.

None of those is an append-only, hash-chained audit ledger with tombstones,
versioning, and W3C PROV-O agents/activities. This module adds exactly that:
one relational table (``provenance_entries``) plus an async manager, and it
JOINS to the graph source-ref system by storing the existing source-ref key on
every ledger entry (``source_ref_key``) instead of inventing a new identifier.

It never touches graph marking, never overwrites ``source_*`` fields, and is
written by a passive, opt-in cognify task (``PROVENANCE_TRACKING``, default
off) that can never break ingestion.
"""

from .manager import ProvenanceBatch, ProvenanceManager, get_provenance_manager
from .models import ProvenanceEntry

__all__ = ["ProvenanceBatch", "ProvenanceManager", "get_provenance_manager", "ProvenanceEntry"]
