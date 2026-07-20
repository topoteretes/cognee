"""Stable encoding for optional dataset-scoped session identifiers.

Cache adapters and the session lifecycle tables historically key sessions by a
single caller-provided ``session_id``.  Dataset-scoped sessions keep those
storage contracts intact by replacing that value with a reversible, versioned
internal identifier at the SessionManager boundary.

An unscoped session is deliberately returned unchanged.  This preserves all
legacy cache keys and, importantly, means scoped reads never need to fall back
to legacy storage that may contain data from more than one dataset.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from uuid import UUID


SESSION_SCOPE_PREFIX = "__cognee_dataset_session_v1__:"


class _InheritDataset:
    __slots__ = ()


# Distinguishes an omitted scope (inherit the manager/current dataset) from an
# explicit ``None`` (force the legacy unscoped namespace).
INHERIT_DATASET = _InheritDataset()
DatasetScopeArg = str | UUID | None | _InheritDataset


@dataclass(frozen=True, slots=True)
class SessionScope:
    """Decoded public session identity and its optional dataset scope."""

    session_id: str
    dataset_id: str | None = None

    @property
    def is_dataset_scoped(self) -> bool:
        return self.dataset_id is not None


def normalize_dataset_id(dataset_id: str | UUID | None) -> str | None:
    """Return the stable string representation used by session storage."""
    if dataset_id is None:
        return None
    normalized = str(dataset_id).strip()
    if not normalized:
        raise ValueError("dataset_id must be a non-empty string when provided")
    try:
        return str(UUID(normalized))
    except ValueError:
        # Legacy database contexts may still carry a dataset name. Public API
        # paths resolve names to UUIDs before reaching session storage.
        pass
    return normalized


def _encode_component(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_component(value: str) -> str:
    if not value:
        raise ValueError("empty scoped-session component")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as error:
        raise ValueError("invalid scoped-session component") from error
    # Reject non-canonical encodings so the SQL prefix predicate and Python
    # decoder always agree about the identity of a scoped row.
    if _encode_component(decoded) != value:
        raise ValueError("non-canonical scoped-session component")
    return decoded


def scoped_session_prefix(dataset_id: str | UUID) -> str:
    """Return the internal-ID prefix unique to one dataset."""
    normalized = normalize_dataset_id(dataset_id)
    if normalized is None:  # pragma: no cover - guarded by the non-optional type
        raise ValueError("dataset_id is required for a scoped session prefix")
    return f"{SESSION_SCOPE_PREFIX}{_encode_component(normalized)}:"


def get_storage_session_id(session_id: str, dataset_id: str | UUID | None = None) -> str:
    """Map a public session id to its cache/lifecycle storage id.

    ``dataset_id=None`` is the legacy namespace and intentionally returns the
    input unchanged.  A dataset produces a deterministic, reversible id.
    """
    public_session_id = str(session_id)
    normalized_dataset_id = normalize_dataset_id(dataset_id)
    if normalized_dataset_id is None:
        if public_session_id.startswith(SESSION_SCOPE_PREFIX):
            raise ValueError(
                f"Unscoped session IDs cannot start with reserved prefix {SESSION_SCOPE_PREFIX!r}"
            )
        return public_session_id
    return f"{scoped_session_prefix(normalized_dataset_id)}{_encode_component(public_session_id)}"


def parse_storage_session_id(storage_session_id: str) -> SessionScope:
    """Decode an internal id, treating malformed/reserved legacy ids as unscoped."""
    value = str(storage_session_id)
    if not value.startswith(SESSION_SCOPE_PREFIX):
        return SessionScope(session_id=value)

    encoded = value[len(SESSION_SCOPE_PREFIX) :]
    try:
        encoded_dataset_id, encoded_session_id = encoded.split(":", 1)
        dataset_id = _decode_component(encoded_dataset_id)
        session_id = _decode_component(encoded_session_id)
    except ValueError:
        # Session IDs were historically unrestricted strings.  A legacy caller
        # may therefore have used our future reserved prefix; malformed values
        # must remain owner-only legacy sessions rather than becoming shareable.
        return SessionScope(session_id=value)

    return SessionScope(session_id=session_id, dataset_id=dataset_id)


def is_dataset_scoped_session_id(storage_session_id: str, dataset_id=None) -> bool:
    """Return whether an internal id is valid and optionally matches a dataset."""
    scope = parse_storage_session_id(storage_session_id)
    if not scope.is_dataset_scoped:
        return False
    expected_dataset_id = normalize_dataset_id(dataset_id)
    return expected_dataset_id is None or scope.dataset_id == expected_dataset_id
