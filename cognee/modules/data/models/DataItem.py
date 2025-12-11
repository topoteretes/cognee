"""
DataItem is a dataclass that allows users to provide custom labels for data items
when adding data to Cognee.

This enables better organization and easier identification of data items,
especially for text data where the default name is just the content hash.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DataItem:
    """
    A dataclass for providing data with optional custom labels.

    Attributes:
        data: The actual data to be ingested into Cognee (can be str, file, DataPoint, etc.)
        label: Optional custom label/name for the data item for better human-friendly identification

    Example:
        >>> item = DataItem(data="Sample text content", label="My Custom Label")
        >>> # Or with a file:
        >>> with open("document.pdf", "rb") as f:
        ...     item = DataItem(data=f, label="Important Document")
    """

    data: Any
    label: Optional[str] = None
