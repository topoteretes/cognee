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
pipeline that stores DataPoints is covered. It records the data item a batch
was stored for, one row holding ids and the item's content hash, and never
the nodes and edges the item produced: the graph's source refs already say
which data item each came from, and a row per node and edge costs about ten
times the source text. The row gets a new version when the item's content
hash changes (an update). Deleting the item retracts it: the paths that
remove its ``Data`` row call ``tombstones.py`` last, once the row is gone,
which writes an ``invalidate`` tombstone on the data item's row (never a row
delete). Dropping only the item's memory keeps the document and writes none;
a cognify rollback retracts the items it gave memory to under its own reason.
A later re-ingest brings a retracted item back, tombstone kept in its version
history. A write or a tombstone that fails raises, so the ingestion or the
delete fails with it instead of leaving a gap nothing reports.

Reading it: ``ProvenanceManager`` (``get_lineage`` / ``revision_history`` /
``verify_chain`` / ``check`` / ``get_statistics`` / ``export``, all
dataset-scopable), ``cognee-cli provenance`` and ``/api/v1/provenance``.
``anchors.py`` adds external HMAC anchors of the chain head
(``PROVENANCE_ANCHOR_KEY`` / ``PROVENANCE_ANCHOR_PATH``) — the defense against
a ledger that is rewritten and re-chained in place.

Three env flags govern three of these systems and must not be mixed up:

- ``COGNEE_PROVENANCE_MODE`` (lightweight|deep|disabled, default lightweight)
  — how much ``source_*`` attribution is stamped onto every stored node.
- ``PROVENANCE_TRACKING`` (default false) — this audit ledger.
- ``EDGE_EVIDENCE_ENABLED`` (default true) — the edge→chunk evidence sidecar.
"""

from .manager import ProvenanceBatch, ProvenanceManager, get_provenance_manager
from .models import ProvenanceEntry

__all__ = ["ProvenanceBatch", "ProvenanceEntry", "ProvenanceManager", "get_provenance_manager"]
