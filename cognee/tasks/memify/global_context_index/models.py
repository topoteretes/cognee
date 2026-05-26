from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


class GlobalContextSummaryContent(BaseModel):
    summary: str


@dataclass
class SummaryNode:
    id: str
    text: str
    type: str
    level: int | None = None
    is_root: bool = False
    dataset_id: str | None = None
    global_context_bucket_id: str | None = None
    child_ids: set[str] = field(default_factory=set)


@dataclass
class GlobalContextIndexInput:
    text_summaries: list[SummaryNode]
    buckets: list[SummaryNode]
    root: SummaryNode | None = None


@dataclass
class BucketAssignment:
    summary_id: str
    bucket_id: str
