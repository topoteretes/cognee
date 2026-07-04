import hashlib
from cognee.modules.governance.models import AuditRecord

HASH_FIELD_ORDER = [
    "actor_id", "action", "target_dataset_id", "outcome",
    "timestamp", "denial_reason", "previous_hash"
]

def compute_row_hash(record_fields: dict[str, str | None]) -> str:
    """
    Computes the SHA-256 hash for one audit record.
    Fields are concatenated in HASH_FIELD_ORDER as UTF-8 strings.
    None values are replaced with the empty string.
    Returns a 64-character lowercase hex string.
    """
    raw = "".join(
        str(record_fields.get(f) or "") for f in HASH_FIELD_ORDER
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class GovernanceBundleIntegrityError(Exception):
    """Raised when a governance bundle fails tamper verification."""
    def __init__(self, record_index: int, field: str, expected: str, got: str):
        self.record_index = record_index
        self.field = field
        super().__init__(
            f"Hash chain broken at record {record_index}: "
            f"{field} expected={expected[:16]}... got={got[:16]}..."
        )

def verify_hash_chain(records: list[AuditRecord]) -> None:
    """
    Verifies that a sequence of AuditRecords forms an intact hash chain.
    Raises GovernanceBundleIntegrityError if:
      - any row_hash does not match the recomputed hash from the record fields
      - any previous_hash does not match the prior record's row_hash
      - the first record's previous_hash is not None
    """
    if not records:
        return

    if records[0].previous_hash is not None and records[0].previous_hash != "":
        raise GovernanceBundleIntegrityError(0, "previous_hash", "None", records[0].previous_hash)

    last_hash = None
    for i, record in enumerate(records):
        timestamp_str = record.timestamp
        from datetime import datetime, timezone
        if isinstance(timestamp_str, datetime):
            if timestamp_str.tzinfo is None:
                timestamp_str = timestamp_str.replace(tzinfo=timezone.utc)
            timestamp_str = timestamp_str.isoformat()
        else:
            timestamp_str = str(timestamp_str)
            
        fields = {
            "actor_id": str(record.actor_id) if record.actor_id else "",
            "action": record.action,
            "target_dataset_id": str(record.target_dataset_id) if record.target_dataset_id else "",
            "outcome": record.outcome,
            "timestamp": timestamp_str,
            "denial_reason": record.denial_reason or "",
            "previous_hash": record.previous_hash or "",
        }
        
        computed = compute_row_hash(fields)
        if computed != record.row_hash:
            raise GovernanceBundleIntegrityError(i, "row_hash", computed, record.row_hash)
            
        if i > 0 and record.previous_hash != last_hash:
            raise GovernanceBundleIntegrityError(i, "previous_hash", str(last_hash), str(record.previous_hash))
            
        last_hash = record.row_hash
