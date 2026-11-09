"""Import receipts (OMP draft v0.6, 3.5.4).

An import SHOULD produce a receipt that lists, per memory, whether it was
accepted, transformed, omitted or rejected, with reasons and any identifier
mappings. :class:`ImportReceipt` is that record for one import run: it is
filled by the loader and the sources as records pass through, and returned
on the import summary.

Bounded on purpose: imports can carry millions of records, so accepted
records are counted (their origin-to-local identifier mappings kept up to a
cap), and only the records that were not simply accepted get an entry (also
capped). Everything dropped past a cap is counted, never silently lost.
"""

from dataclasses import dataclass, field
from typing import Any

RECEIPT_STATUSES = ("accepted", "transformed", "omitted", "rejected")
MAX_RECEIPT_ENTRIES = 1000
MAX_IDENTIFIER_MAPPINGS = 10_000


@dataclass
class ImportReceipt:
    """What happened to each record of one import.

    ``counts`` tallies records per status. ``entries`` lists the transformed,
    omitted and rejected records with a reason (and the references that
    could not be resolved, for dangling ones). ``identifier_map`` keeps the
    origin identifier -> local id mapping the receiving system assigned
    (OMP 3.5.2) wherever the two differ. ``integrity`` reports checksum
    verification: how many verified, how many carried no checksum, and the
    identifiers whose checksum did not match (reported, never rejected:
    OMP 3.5.7 treats integrity as a check, not a gate).
    """

    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RECEIPT_STATUSES, 0))
    entries: list[dict[str, Any]] = field(default_factory=list)
    identifier_map: dict[str, str] = field(default_factory=dict)
    integrity: dict[str, Any] = field(
        default_factory=lambda: {"verified": 0, "unverified": 0, "failed": []}
    )
    entries_dropped: int = 0
    mappings_dropped: int = 0

    def record(
        self,
        status: str,
        identifier: str,
        *,
        kind: str | None = None,
        reason: str | None = None,
        local_id: Any = None,
        refs: list[str] | None = None,
    ) -> None:
        """Note one record's outcome.

        ``local_id`` is the id the record got in this store; it is kept as a
        mapping when it differs from the origin identifier. ``refs`` are the
        references a dangling record could not resolve (OMP 3.5.5: kept and
        marked, never dropped silently).
        """
        if status not in RECEIPT_STATUSES:
            raise ValueError(
                f"Unknown receipt status {status!r}. Expected one of {RECEIPT_STATUSES}."
            )
        self.counts[status] += 1
        if local_id is not None and str(local_id) != identifier:
            if len(self.identifier_map) < MAX_IDENTIFIER_MAPPINGS:
                self.identifier_map[identifier] = str(local_id)
            else:
                self.mappings_dropped += 1
        if status == "accepted":
            return
        entry: dict[str, Any] = {"status": status, "identifier": identifier}
        if kind:
            entry["kind"] = kind
        if reason:
            entry["reason"] = reason
        if local_id is not None:
            entry["local_id"] = str(local_id)
        if refs:
            entry["refs"] = list(refs)
        if len(self.entries) < MAX_RECEIPT_ENTRIES:
            self.entries.append(entry)
        else:
            self.entries_dropped += 1

    def integrity_check(self, identifier: str, verified: bool | None) -> None:
        """Note a checksum verification: True (matched), False (did not match)
        or None (the record carried no checksum)."""
        if verified is None:
            self.integrity["unverified"] += 1
        elif verified:
            self.integrity["verified"] += 1
        else:
            failed = self.integrity["failed"]
            if len(failed) < MAX_RECEIPT_ENTRIES:
                failed.append(identifier)
            self.integrity["failed_count"] = self.integrity.get("failed_count", 0) + 1

    @property
    def dangling(self) -> list[dict[str, Any]]:
        """The entries kept because a reference could not be resolved."""
        return [entry for entry in self.entries if entry.get("refs")]

    def summary(self) -> dict[str, Any]:
        """A JSON-safe snapshot for the import summary."""
        return {
            "counts": dict(self.counts),
            "entries": list(self.entries),
            "entries_dropped": self.entries_dropped,
            "identifier_map": dict(self.identifier_map),
            "mappings_dropped": self.mappings_dropped,
            "integrity": {
                key: (list(value) if isinstance(value, list) else value)
                for key, value in self.integrity.items()
            },
        }
