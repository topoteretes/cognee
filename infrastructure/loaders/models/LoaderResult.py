from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum


class ContentType(Enum):
    """Content type classification for loaded files"""

    TEXT = "text"
    STRUCTURED = "structured"
    BINARY = "binary"


class LoaderResult(BaseModel):
    """
    Standardized output format for all file loaders.

    This model ensures consistent data structure across all loader implementations,
    following cognee's pattern of using Pydantic models for data validation.
    """

    content: str  # Primary text content extracted from file
    metadata: Dict[str, Any]  # File metadata (name, size, type, loader info, etc.)
    content_type: ContentType  # Content classification
    chunks: Optional[List[str]] = None  # Pre-chunked content if available
    source_info: Optional[Dict[str, Any]] = None  # Source-specific information

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the loader result to a dictionary format.

        Returns:
            Dict containing all loader result data with string-serialized content_type
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "content_type": self.content_type.value,
            "source_info": self.source_info or {},
            "chunks": self.chunks,
        }

    class Config:
        """Pydantic configuration following cognee patterns"""

        use_enum_values = True
        validate_assignment = True
