from typing import Any
from cognee.modules.data.models import Data

def extract_data_info(data: Any) -> Any:
    if not data:
        return "None"
    elif isinstance(data, list) and all(isinstance(item, Data) for item in data):
        return [str(item.id) for item in data]
    else:
        # Cap stringified payload to prevent unbounded pipeline_runs growth
        data_str = str(data)
        max_len = 250
        if len(data_str) > max_len:
            return f"{data_str[:max_len]}... (truncated from {len(data_str)} chars)"
        return data_str

