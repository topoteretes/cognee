"""Edge evidence: which chunk of which document supports a graph edge.

One relational table (``provenance_edge_evidence``) links a graph edge id to
the ``DocumentChunk``, ``Data`` revision, and ``PipelineRun`` that produced it.
Retrieval reads it to attach ``EvidenceReference`` citations to answers.

Flags (see ``.env.template``, "Provenance" section):

- ``EDGE_EVIDENCE_ENABLED`` (default true) — capture on/off.
- ``EDGE_EVIDENCE_FLUSH_THRESHOLD`` (default 10000) — pending rows per data
  item before an early bulk flush; otherwise one flush per data item.

Not to be confused with ``COGNEE_PROVENANCE_MODE`` (source_* stamps written
onto nodes) or ``PROVENANCE_TRACKING`` (the hash-chained audit ledger in the
parent package). Rows here are cheap observations, not audit records: they are
bulk-inserted without chaining, ignored at read time when their pipeline run
did not complete or their document is gone, and swept by ``cleanup`` when the
document or its memory is deleted.

Scope: evidence is captured only where document chunks are stored
(``add_data_points`` during cognify). Edges from contradiction detection,
``improve()`` enrichment, session bridging, and the code-graph route carry no
evidence rows yet; ``evidence_kind`` is the hook for adding them.
"""

from .cleanup import delete_edge_evidence
from .config import ProvenanceConfig, get_provenance_config
from .models import ProvenanceEdgeEvidence

__all__ = [
    "ProvenanceConfig",
    "ProvenanceEdgeEvidence",
    "delete_edge_evidence",
    "get_provenance_config",
]
