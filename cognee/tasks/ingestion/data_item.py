from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None

