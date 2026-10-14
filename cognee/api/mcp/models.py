"""Pydantic data models and schemas for Cognee FastMCP Server."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchTypeEnum(str, Enum):
    """Supported Cognee search types."""

    INSIGHTS = "INSIGHTS"
    CHUNKS = "CHUNKS"
    SUMMARIES = "SUMMARIES"
    COMPLETION = "COMPLETION"

    @classmethod
    def from_str(cls, value: str) -> "SearchTypeEnum":
        """Parse case-insensitively into SearchTypeEnum, defaulting to INSIGHTS."""
        normalized = value.strip().upper()
        for member in cls:
            if member.value == normalized or member.name == normalized:
                return member
        return cls.INSIGHTS


class AddMemoryRequest(BaseModel):
    """Schema for adding memory into Cognee."""

    content: str = Field(
        ...,
        min_length=1,
        description="Text content, note, or document chunk to store in memory."
    )
    dataset_name: str = Field(
        default="default_dataset",
        description="Target dataset or namespace in the Cognee knowledge graph."
    )
    auto_cognify: bool = Field(
        default=True,
        description="Whether to run cognify() immediately to generate graph nodes and relationships."
    )


class AddMemoryResponse(BaseModel):
    """Structured response after ingesting memory."""

    success: bool = Field(description="True if ingestion and optional cognification succeeded")
    dataset_name: str = Field(description="The dataset where content was stored")
    message: str = Field(description="Human-readable result summary")
    cognified: bool = Field(description="Whether cognify() was executed")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class SearchMemoryRequest(BaseModel):
    """Schema for searching Cognee memory."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query or question to retrieve context for."
    )
    search_type: SearchTypeEnum = Field(
        default=SearchTypeEnum.INSIGHTS,
        description="Search mode: INSIGHTS (graph insights), CHUNKS (raw vectors), SUMMARIES, or COMPLETION."
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Optional dataset filter. If omitted, searches default/all datasets."
    )


class SearchMemoryResponse(BaseModel):
    """Structured response after memory search."""

    success: bool = Field(description="True if search executed successfully")
    query: str = Field(description="Original search query")
    search_type: str = Field(description="Applied search mode")
    dataset_name: Optional[str] = Field(default=None, description="Filtered dataset name if any")
    total_results: int = Field(default=0, description="Number of items returned")
    results: List[Any] = Field(default_factory=list, description="Retrieved context items, chunks, or graph insights")
    error: Optional[str] = Field(default=None, description="Error message if search failed")


class PruneMemoryResponse(BaseModel):
    """Structured response after pruning memory."""

    success: bool = Field(description="True if prune completed")
    message: str = Field(description="Summary message")
    prune_all: bool = Field(description="Whether entire system was pruned")
    error: Optional[str] = Field(default=None, description="Error message if prune failed")
