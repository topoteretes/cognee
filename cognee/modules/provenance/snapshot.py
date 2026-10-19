"""Compact mutation snapshots for ledger rows.

A ledger row records *that* an entity was asserted, re-asserted or retracted;
the snapshot records *what* was asserted, compactly enough to store on every
row: the DataPoint's semantic properties (long text reduced to a hash and a
length) and one SHA-256 over them. It lives under ``metadata["snapshot"]`` so
it is covered by the row checksum without changing the hash scheme — adding
a column to ``integrity._HASHED_FIELDS`` would invalidate every existing
ledger.

Three things fall out of it:

- **Deltas.** When an entity is re-asserted, the manager stores the field
  level difference against the previous version under ``metadata["delta"]``,
  so ``revision_history`` reads "description changed from X to Y" instead of
  "edited".
- **No-op suppression.** A re-mention whose snapshot hash equals the live
  row's, from the same document, is not a new version. Re-mentions across
  chunks of one document used to version the entity every time.
- **Drift detection.** ``drift.check_drift`` re-reads each live node from
  the graph, re-hashes the same fields, and reports rows whose graph state no
  longer matches what the ledger last recorded.

What is snapshotted: every pydantic field of the DataPoint except identity,
bookkeeping, attribution and tunable-weight fields (``EXCLUDED_FIELDS``), and
except nested DataPoints / edges (those are rows of their own). Values are
normalized through JSON so a DataPoint in memory and the same node read back
from a graph adapter hash identically.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID

# Identity, bookkeeping, attribution and weights: none of these are the
# entity's content. ``feedback_weight`` / ``importance_weight`` are tuned by
# improve(); including them would report every feedback pass as drift.
EXCLUDED_FIELDS = frozenset(
    {
        "id",
        "created_at",
        "updated_at",
        "version",
        "topological_rank",
        "metadata",
        "belongs_to_set",
        "source_pipeline",
        "source_task",
        "source_node_set",
        "source_user",
        "source_content_hash",
        "feedback_weight",
        "importance_weight",
        "pipeline_run_id",
    }
)

# Strings longer than this are stored as {"sha256", "len"}: chunk text and
# long descriptions must not bloat the relational ledger.
MAX_INLINE_CHARS = 256


def _normalize(value: Any) -> Any:
    """JSON-stable, compact form of one field value; ``...`` marks "skip"."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        if len(value) > MAX_INLINE_CHARS:
            return {
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "len": len(value),
            }
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_normalize(item) for item in value]
        if any(item is ... for item in items):
            return ...
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            return ...
        inner = {key: _normalize(item) for key, item in value.items()}
        if any(item is ... for item in inner.values()):
            return ...
        return inner
    # Nested DataPoints, Edge objects, arbitrary models: a row of their own.
    return ...


def _is_data_point(value: Any) -> bool:
    return hasattr(value, "model_fields") and hasattr(value, "id") and hasattr(value, "metadata")


def snapshot_fields(data_point: Any) -> dict[str, Any]:
    """The snapshot-able fields of a DataPoint, normalized."""
    model_fields = getattr(type(data_point), "model_fields", None) or {}
    fields: dict[str, Any] = {}
    for name in model_fields:
        if name in EXCLUDED_FIELDS:
            continue
        value = getattr(data_point, name, None)
        if _is_data_point(value) or (
            isinstance(value, (list, tuple)) and any(_is_data_point(item) for item in value)
        ):
            continue
        normalized = _normalize(value)
        if normalized is ...:
            continue
        fields[name] = normalized
    return fields


def snapshot_from_mapping(
    node: Mapping[str, Any], recorded_fields: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-project a graph node dict onto the field set a snapshot recorded.

    Adapters return extra properties (labels, stamps, JSON blobs); using the
    recorded field names keeps the comparison to what the ledger hashed.
    A field missing from the node is recorded as ``None``. Adapters store
    non-scalar properties as JSON strings, so when the ledger recorded a list
    or dict and the node holds a string, the string is decoded first.
    """
    fields: dict[str, Any] = {}
    for name, recorded in recorded_fields.items():
        value = node.get(name)
        if isinstance(value, str) and isinstance(recorded, (list, dict)):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        normalized = _normalize(value)
        fields[name] = None if normalized is ... else normalized
    return fields


def content_hash(fields: Mapping[str, Any]) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_snapshot(data_point: Any) -> dict[str, Any] | None:
    """``{"hash": sha256, "fields": {...}}`` for one DataPoint.

    None for objects that are not pydantic models (nothing to snapshot).
    """
    if not getattr(type(data_point), "model_fields", None):
        return None
    fields = snapshot_fields(data_point)
    return {"hash": content_hash(fields), "fields": fields}


def snapshot_metadata(data_point: Any) -> dict[str, Any]:
    """``{"snapshot": ...}`` to splice into a track call's metadata, or ``{}``."""
    snapshot = build_snapshot(data_point)
    return {"snapshot": snapshot} if snapshot else {}


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
