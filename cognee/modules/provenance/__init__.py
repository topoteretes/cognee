"""Audit-grade provenance ledger (append-only, tamper-evident).

This module is the fifth — and only audit-grade — provenance system in cognee.
It is deliberately distinct from the other four:

- ``cognee/infrastructure/databases/provenance/`` — graph source-refs used for
  delete/rollback ownership (``source_ref:v1:{dataset_id}:{data_id}`` keys).
- ``source_*`` DataPoint stamping — write-time attribution on graph nodes.
- the memory-provenance projection — actor-centric views over session memory.
- ``cognee/modules/provenance/edge_evidence/`` — assertion-level edge→chunk
  evidence sidecar (``provenance_edge_evidence``), linking graph edges to the
  chunks, data revisions, and pipeline runs that produced them.

None of those is an append-only, hash-chained audit ledger with tombstones,
versioning, and W3C PROV-O agents/activities. This module adds exactly that:
one relational table (``provenance_entries``) plus an async manager, and it
JOINS to the graph source-ref system by storing the existing source-ref key on
every ledger entry (``source_ref_key``) instead of inventing a new identifier.

It never touches graph marking, never overwrites ``source_*`` fields, and is
written passively at the ``add_data_points`` storage seam (opt-in via
``PROVENANCE_TRACKING``, default off; ``cognee/tasks/provenance/``) so every
pipeline that stores DataPoints is covered, and it can never break ingestion.
Deletions reach it the same way: the graph delete choke points, cognify
rollback and incremental update call ``tombstones.py``, which turns
hard-deleted elements into ``invalidate`` tombstones (never row deletes) and
can never break a delete. A later re-ingest of the same content resurrects
them, tombstone kept in the version history.

Reading it: ``ProvenanceManager`` (``get_lineage`` / ``revision_history`` /
``verify_chain`` / ``check`` / ``get_statistics`` / ``export``, all
dataset-scopable), ``cognee-cli provenance`` and ``/api/v1/provenance``.
``anchors.py`` adds external HMAC anchors of the chain head
(``PROVENANCE_ANCHOR_KEY`` / ``PROVENANCE_ANCHOR_PATH``) — the defense against
a ledger that is rewritten and re-chained in place. ``snapshot.py`` puts a
compact content snapshot on every node row (field deltas between versions,
no-op suppression of identical re-mentions) and ``drift.py`` compares those
snapshots with the live graph — the defense against a graph edited behind
the ledger's back.

Three env flags govern three of these systems and must not be mixed up:

- ``COGNEE_PROVENANCE_MODE`` (lightweight|deep|disabled, default lightweight)
  — how much ``source_*`` attribution is stamped onto every stored node.
- ``PROVENANCE_TRACKING`` (default false) — this audit ledger.
- ``EDGE_EVIDENCE_ENABLED`` (default true) — the edge→chunk evidence sidecar.
"""

from .manager import ProvenanceBatch, ProvenanceManager, get_provenance_manager
from .models import ProvenanceEntry

__all__ = ["ProvenanceBatch", "ProvenanceEntry", "ProvenanceManager", "get_provenance_manager"]
