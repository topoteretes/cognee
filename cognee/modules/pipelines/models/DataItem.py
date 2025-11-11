from dataclasses import dataclass
from typing import Optional
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

@dataclass
class DataItem:
    id: str
    name: str
    source: str
    status: DataItemStatus
    label: Optional[str] = None  # new field to add a label
