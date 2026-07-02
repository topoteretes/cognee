"""Domain models for the Support-Triage Bot.

Pydantic models representing support threads, citations, and triage results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SupportThread(BaseModel):
    """A resolved support thread ready for ingestion into cognee memory."""

    thread_id: str = Field(..., description="Platform-specific thread identifier (e.g. Slack thread_ts)")
    channel_id: str = Field(..., description="Channel where the thread lives")
    reporter: str = Field(default="unknown", description="User who reported the issue")
    problem_summary: str = Field(..., description="One-line summary of the problem")
    resolution_summary: str = Field(..., description="How the issue was resolved")
    messages: list[str] = Field(default_factory=list, description="Ordered messages in the thread")
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    thread_url: str = Field(default="", description="Permalink to the thread")

    def to_document(self) -> str:
        """Render the thread as a structured document for cognee.remember()."""
        parts = [
            f"## Support Thread: {self.problem_summary}",
            "",
            f"**Thread ID:** {self.thread_id}",
            f"**Channel:** {self.channel_id}",
            f"**Reporter:** {self.reporter}",
            f"**Resolved:** {self.resolved_at.isoformat()}",
            f"**URL:** {self.thread_url}" if self.thread_url else "",
            "",
            "### Problem",
            self.problem_summary,
            "",
            "### Resolution",
            self.resolution_summary,
            "",
            "### Conversation",
        ]
        for i, msg in enumerate(self.messages, 1):
            parts.append(f"{i}. {msg}")
        return "\n".join(parts)


class Citation(BaseModel):
    """A ranked citation from a recalled past issue."""

    source_thread_id: str = Field(..., description="Thread ID of the cited past issue")
    thread_url: str = Field(default="", description="Permalink to the cited thread")
    resolution_summary: str = Field(..., description="How the cited issue was resolved")
    similarity_score: Optional[float] = Field(default=None, description="Recall relevance score")
    resolved_at: Optional[datetime] = Field(default=None, description="When the cited issue was resolved")


class TriageResult(BaseModel):
    """The complete triage output for a new support issue."""

    query: str = Field(..., description="The original support query")
    citations: list[Citation] = Field(default_factory=list, description="Ranked citations from past issues")
    suggested_reply: str = Field(default="", description="LLM-generated suggestion text")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence score")
