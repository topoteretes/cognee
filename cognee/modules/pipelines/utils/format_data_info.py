from typing import Any
from cognee.modules.data.models import Data


def format_data_info(data: Any) -> str | list[str]:
    if not data:
        return "None"
    elif isinstance(data, list) and all(isinstance(item, Data) for item in data):
        return [str(item.id) for item in data]
    else:
        stringified_data = str(data)
        if len(stringified_data) > 500:
            return f"{stringified_data[:500]}... [truncated {len(stringified_data)} chars]"
        return stringified_data
