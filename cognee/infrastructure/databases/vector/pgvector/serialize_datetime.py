from datetime import datetime

def serialize_datetime(data):
    """Recursively convert datetime objects in dictionaries/lists to ISO format."""
    if isinstance(data, dict):
        return {key: serialize_datetime(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_datetime(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()  # Convert datetime to ISO 8601 string
    else:
        return data