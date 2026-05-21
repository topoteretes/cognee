from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID


@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None
    external_metadata: Optional[dict] = field(default=None)
    data_id: Optional[UUID] = None
