from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None
