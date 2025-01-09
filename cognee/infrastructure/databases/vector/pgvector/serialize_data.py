from datetime import datetime
from uuid import UUID


def serialize_data(data):
    """Recursively convert datetime objects in dictionaries/lists to ISO format."""
    if isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_data(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()  # Convert datetime to ISO 8601 string
    elif isinstance(data, UUID):
        return str(data)
    else:
        return data
