"""Snapshots on ledger rows: what a row recorded, and how two versions differ.

A row may carry ``metadata["snapshot"] = {"fields": {...}, "hash": ...}``. It
lives inside ``metadata`` so the row checksum covers it without changing the
hash scheme: adding a column to ``integrity._HASHED_FIELDS`` would invalidate
every existing ledger.

The data item row uses it for the item's content hash
(``ProvenanceBatch.track_data_item``). Two things follow in the manager
(``_mutation_decision``):

- **No-op suppression.** Re-tracking a live row with an equal snapshot is not
  a new version, so every batch stored for a data item can queue its row.
- **Deltas.** When the snapshot differs, the new version stores the field
  level difference under ``metadata["delta"]``, so ``revision_history`` reads
  "content hash changed from X to Y" instead of "edited".
"""

from collections.abc import Mapping
from typing import Any


def diff_snapshots(
    old_fields: Mapping[str, Any] | None, new_fields: Mapping[str, Any] | None
) -> dict[str, list[Any]]:
    """``{field: [old, new]}`` for every field whose value differs."""
    old_fields = old_fields or {}
    new_fields = new_fields or {}
    delta: dict[str, list[Any]] = {}
    for name in sorted(set(old_fields) | set(new_fields)):
        old_value = old_fields.get(name)
        new_value = new_fields.get(name)
        if old_value != new_value:
            delta[name] = [old_value, new_value]
    return delta
