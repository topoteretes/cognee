"""External anchoring of the ledger hash chain.

A hash chain inside one database proves nothing against an attacker who can
rewrite rows *and* recompute every checksum after them: the rewritten ledger
is internally consistent. Anchoring fixes that by periodically recording the
chain head — ``(sequence_id, checksum)`` — somewhere the database writer
cannot forge: signed with an HMAC key the database never holds
(``PROVENANCE_ANCHOR_KEY``) and appended to a file outside the database
(``PROVENANCE_ANCHOR_PATH``, default ``{system_root}/provenance_anchors.jsonl``).

Verification replays each anchor: the signature must verify with the current
key, and the ledger row at ``sequence_id`` must still carry the anchored
checksum. A rewrite that re-chains the ledger changes that checksum, and the
attacker cannot produce a matching anchor without the key. Anchors can be
shipped elsewhere too (object storage, a notary, a ticket) — the record is a
plain dict and ``verify_anchor`` takes any list of them.

The anchor file is append-only JSONL. ``PROVENANCE_ANCHOR_PATH`` should point
somewhere the database host cannot write; keeping it next to the SQLite file
protects against SQL-level tampering only.
"""

import hashlib
import hmac
import json
import os
from typing import Any

from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.logging_utils import get_logger

from . import storage
from .models.ProvenanceEntry import utc_now_iso

logger = get_logger("provenance.anchors")

_SEP = "\x1f"
ANCHOR_VERSION = 1


class AnchoringNotConfiguredError(RuntimeError):
    """Raised when an anchor is requested without ``PROVENANCE_ANCHOR_KEY``."""


def anchoring_configured() -> bool:
    return bool(get_cognify_config().provenance_anchor_key)


def _key() -> bytes:
    key = get_cognify_config().provenance_anchor_key
    if not key:
        raise AnchoringNotConfiguredError(
            "Set PROVENANCE_ANCHOR_KEY to anchor or verify anchors of the provenance ledger."
        )
    return key.encode("utf-8")


def key_id() -> str:
    """Short, non-reversible identifier of the configured key (rotation aid)."""
    return hashlib.sha256(_key()).hexdigest()[:12]


def anchor_path() -> str:
    configured = get_cognify_config().provenance_anchor_path
    if configured:
        return configured
    return os.path.join(get_base_config().system_root_directory, "provenance_anchors.jsonl")


def _payload(sequence_id: int, checksum: str, anchored_at: str, total_entries: int) -> bytes:
    return _SEP.join(
        [str(ANCHOR_VERSION), str(sequence_id), checksum, anchored_at, str(total_entries)]
    ).encode("utf-8")


def sign_anchor(anchor: dict[str, Any]) -> str:
    return hmac.new(
        _key(),
        _payload(
            anchor["sequence_id"],
            anchor["checksum"],
            anchor["anchored_at"],
            anchor["total_entries"],
        ),
        hashlib.sha256,
    ).hexdigest()


def build_anchor(sequence_id: int, checksum: str, total_entries: int) -> dict[str, Any]:
    anchor = {
        "version": ANCHOR_VERSION,
        "sequence_id": sequence_id,
        "checksum": checksum,
        "total_entries": total_entries,
        "anchored_at": utc_now_iso(),
        "key_id": key_id(),
    }
    anchor["signature"] = sign_anchor(anchor)
    return anchor


async def anchor_chain_head() -> dict[str, Any] | None:
    """Sign the current chain head and append it to the anchor file.

    Returns the anchor, or None when the ledger is empty (nothing to anchor).
    """
    _key()  # fail fast on a missing key, before touching the ledger
    async with storage.get_async_session() as session:
        head = await storage.get_chain_head(session)
    if head is None:
        return None
    sequence_id, checksum = head
    if not checksum:
        return None
    stats = await storage.aggregate_statistics()
    anchor = build_anchor(sequence_id, checksum, stats["total_entries"])

    path = anchor_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(anchor, sort_keys=True) + "\n")
    logger.info("Anchored provenance chain head #%s to %s", sequence_id, path)
    return anchor


def read_anchors(path: str | None = None) -> list[dict[str, Any]]:
    """Every anchor in the anchor file, in the order they were written."""
    path = path or anchor_path()
    if not os.path.exists(path):
        return []
    anchors: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                anchors.append(json.loads(line))
            except json.JSONDecodeError:
                anchors.append({"_malformed": True, "line": line_number})
    return anchors


def signature_valid(anchor: dict[str, Any]) -> bool:
    try:
        return hmac.compare_digest(sign_anchor(anchor), str(anchor.get("signature", "")))
    except (KeyError, TypeError, ValueError):
        return False


async def verify_anchors(anchors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Check every anchor's signature and that the ledger still matches it.

    An anchor fails when its signature does not verify (``bad_signature``),
    when the ledger has no row at its position (``missing_row`` — rows were
    deleted or the ledger was reset), or when the row's checksum differs
    (``checksum_mismatch`` — the ledger was rewritten and re-chained after
    the anchor). ``valid`` is False when any anchor fails; ``anchored`` is
    False when there are no anchors at all, which is a gap, not a failure.
    """
    anchors = read_anchors() if anchors is None else anchors
    if not anchors:
        return {"valid": True, "anchored": False, "anchors_checked": 0, "failures": []}

    positions = [a["sequence_id"] for a in anchors if isinstance(a.get("sequence_id"), int)]
    stored = await storage.retrieve_checksums_by_sequence(positions)

    failures: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor.get("_malformed"):
            failures.append({"line": anchor.get("line"), "reason": "malformed"})
            continue
        if not signature_valid(anchor):
            failures.append({"sequence_id": anchor.get("sequence_id"), "reason": "bad_signature"})
            continue
        sequence_id = anchor["sequence_id"]
        if sequence_id not in stored:
            failures.append({"sequence_id": sequence_id, "reason": "missing_row"})
        elif stored[sequence_id] != anchor["checksum"]:
            failures.append(
                {
                    "sequence_id": sequence_id,
                    "reason": "checksum_mismatch",
                    "anchored_checksum": anchor["checksum"],
                    "stored_checksum": stored[sequence_id],
                }
            )

    latest = max(
        (a for a in anchors if isinstance(a.get("sequence_id"), int)),
        key=lambda a: a["sequence_id"],
        default=None,
    )
    return {
        "valid": not failures,
        "anchored": True,
        "anchors_checked": len(anchors),
        "latest_anchor": latest,
        "failures": failures,
    }
