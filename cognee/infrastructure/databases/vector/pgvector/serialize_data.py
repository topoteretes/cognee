from datetime import datetime
from uuid import UUID


def serialize_data(data):
    """
    Recursively convert various data types to serializable formats.

    This function processes dictionaries and lists, converting any datetime objects to ISO
    8601 strings, and UUID objects to their string representation. Other data types are
    returned unchanged. It handles recursive structures if present.

    Parameters:
    -----------

        - data: The input data to serialize, which can be a dict, list, datetime, UUID, or
          other types.

    Returns:
    --------

        The serialized representation of the input data, with datetime objects converted to
        ISO format and UUIDs to strings.
    """
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
