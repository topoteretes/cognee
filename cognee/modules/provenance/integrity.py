"""Integrity primitives for the provenance ledger (pure, sync).

Checksum canonicalization is a deliberate break from semantica's byte format,
which had two defects: no-separator concatenation (field-boundary ambiguity)
and ``None`` hashing differently on the object vs dict path. Cognee ships no
semantica databases to migrate, so both are fixed here:

- every field value is canonicalized as compact, sorted-key JSON (so raw
  separator bytes inside a value, ``None`` vs ``""``, and list element
  boundaries are all unambiguous), and
- the JSON parts are joined with the ``\\x1f`` unit separator (JSON strings
  escape control characters, so the separator can never appear inside a part).

``_HASHED_FIELDS`` covers EVERY column of the ledger row except ``checksum``
itself (the hash output) and the raw ``entity_id``. Order is a frozen
contract — changing it (or the canonicalization rules) invalidates every
stored checksum.

The raw ``entity_id`` is not hashed directly because the manager's versioning
archives a prior value by relabeling it to a new entity_id
(``"X" -> "X:v:{last_updated}[:{n}]"``); hashing the raw id would change the
archived copy's checksum and falsely break any later entry whose
``previous_checksum`` had already chained from the pre-relabel value. Instead
the CANONICAL entity id is hashed: the archive suffix — recognizable because
it embeds the row's own ``last_updated`` — is stripped first, so a relabel is
checksum-stable while swapping the entity_ids of two rows (reattributing an
audit trail) still breaks verification.
"""

import hashlib
import json
from typing import Any, Optional

# Order is the contract. Every ProvenanceEntry field except ``checksum`` (the
# hash output) and ``entity_id`` (hashed in canonical form, see module docstring).
_HASHED_FIELDS = (
    "entity_type",
    "activity_id",
    "agent_id",
    "agent_type",
    "is_automated",
    "role",
    "source_document",
    "source_location",
    "source_quote",
    "source_ref_key",
    "timestamp",
    "first_seen",
    "last_updated",
    "activity_started_at_time",
    "activity_ended_at_time",
    "valid_from",
    "valid_until",
    "confidence",
    "credibility",
    "sequence_id",
    "previous_checksum",
    "parent_entity_id",
    "used_entities",
    "previous_version_id",
    "derived_from_id",
    "acted_on_behalf_of",
    "informed_by_activities",
    "revision_type",
    "supersedes",
    "bundle_id",
    "invalidated",
    "invalidated_at_time",
    "invalidated_by",
    "invalidation_reason",
    "start_index",
    "end_index",
    "metadata",
    "version",
)
_SEP = "\x1f"  # unit separator: unambiguous field boundaries


def canonical_entity_id(entry: Any) -> str:
    """The entity id with any archive-relabel suffix stripped.

    Archive ids are ``"{original}:v:{last_updated}"`` (plus ``":{n}"`` on
    collision) where ``last_updated`` is the archived row's OWN last_updated —
    so the suffix is recognizable from the entry itself and stripping it is
    deterministic. Live entries pass through unchanged.
    """
    entity_id = getattr(entry, "entity_id", None) or ""
    last_updated = getattr(entry, "last_updated", None)
    if last_updated:
        marker = f":v:{last_updated}"
        index = entity_id.rfind(marker)
        if index != -1:
            tail = entity_id[index + len(marker) :]
            if not tail or (tail.startswith(":") and tail[1:].isdigit()):
                return entity_id[:index]
    return entity_id


def _canonical_part(value: Any) -> str:
    """Compact, sorted-key JSON: unambiguous for None/str/bool/num/list/dict."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compute_checksum(entry: Any) -> str:
    """Compute the SHA-256 checksum over the entry's hashed fields.

    Accepts any object exposing the ``_HASHED_FIELDS`` attributes (the Pydantic
    ``ProvenanceEntry`` in practice).
    """
    parts = [_canonical_part(canonical_entity_id(entry))]
    for name in _HASHED_FIELDS:
        parts.append(_canonical_part(getattr(entry, name, None)))
    return hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()


def verify_checksum(entry: Any, expected: Optional[str] = None) -> bool:
    """Recompute the entry's checksum and compare it to the stored/expected one.

    Returns False when no checksum is present to compare against.
    """
    if expected is None:
        expected = getattr(entry, "checksum", None)
    if expected is None:
        return False
    return compute_checksum(entry) == expected
