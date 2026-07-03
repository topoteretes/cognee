"""Dataset versioning built on the COG-5522 run ledger (issue #3650, Approach 1).

The ordered sequence of completed pipeline runs *is* the version history:

- **snapshot** — a named label at a ledger cut (no data copied),
- **as-of reads** — filter the live store to runs completed by T
  (forward-faithful; see ``methods/as_of_read.py`` for the boundary),
- **rollback** — reverse post-T runs with the existing rollback primitive,
  captured as an undoable ledger op,
- **reversible forget / undo** — a write-ahead ledgered inverse restores the
  exact graph rows, provenance, and vector embeddings within the retention
  window (``VERSION_RETENTION_DAYS``, default 30).
"""

from .methods import (
    capture_forget_inverse,
    create_snapshot,
    get_graph_as_of,
    get_version_op,
    get_visible_artifacts_as_of,
    list_snapshots,
    mark_forget_applied,
    resolve_as_of_time,
    rollback_dataset_to,
    search_chunks_as_of,
    undo_version_op,
)
from .models import DatasetSnapshot, VersionOp, VersionOpStatus
