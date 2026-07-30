from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID


@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None
    external_metadata: Optional[dict] = field(default=None)
    data_id: Optional[UUID] = None
    # Optional pre-computed content hash. When set together with an explicit
    # data_id, the add pipeline's incremental skip compares it against the
    # stored Data.content_hash and reprocesses on mismatch — this is how a
    # stable-id item (e.g. a DLT source manifest) signals "same identity, new
    # content". Must use the same formula as ingestion's content hashing
    # (plain md5 over the stored bytes) or changed content goes undetected.
    content_hash: Optional[str] = None
