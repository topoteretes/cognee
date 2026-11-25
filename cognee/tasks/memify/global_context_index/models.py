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
    graph_bucket_entity_ids: set[str] | None = None


@dataclass(init=False)
class GlobalContextIndexUpdateData:
    text_summaries: list[SummaryNode]
    buckets: list[SummaryNode]
    root: SummaryNode | None = None

    def __init__(
        self,
        text_summaries: list[SummaryNode],
        buckets: list[SummaryNode],
        root: SummaryNode | None = None,
        entities_by_summary_id: dict[str, set[str]] | None = None,
    ):
        # Compatibility-only argument from GlobalContextIndexInput; graph evidence is loaded later.
        _ = entities_by_summary_id
        self.text_summaries = text_summaries
        self.buckets = buckets
        self.root = root


GlobalContextIndexInput = GlobalContextIndexUpdateData


@dataclass(init=False)
class BucketAssignment:
    child_id: str
    parent_id: str

    def __init__(
        self,
        child_id: str | None = None,
        parent_id: str | None = None,
        *,
        summary_id: str | None = None,
        bucket_id: str | None = None,
    ):
        resolved_child_id = child_id if child_id is not None else summary_id
        resolved_parent_id = parent_id if parent_id is not None else bucket_id
        if resolved_child_id is None or resolved_parent_id is None:
            raise TypeError("BucketAssignment requires child_id and parent_id.")
        self.child_id = resolved_child_id
        self.parent_id = resolved_parent_id

    @property
    def summary_id(self) -> str:
        return self.child_id

    @property
    def bucket_id(self) -> str:
        return self.parent_id


@dataclass(frozen=True)
class DatasetContent:
    """A dataset's documents, their chunks, the entities those chunks mention, and its
    index as stored now. Every chunk and entity id in it resolves."""

    dataset_id: str
    document_names: dict[str, str]  # document id -> name, for documents that have one
    document_by_chunk: dict[str, str]  # chunk id -> document id
    entities: dict[str, DatasetEntity]  # only entities some chunk mentions
    relations: dict[tuple[str, str], tuple[str, ...]]  # (source id, target id) -> edge texts
    existing_summary_texts: dict[str, str]  # index node id -> text; only the write reads it

    def document_name(self, document_id: str) -> str:
        """A document with no name on its chunks renders as its id."""
        return self.document_names.get(document_id, document_id)


@dataclass(frozen=True)
class DatasetEntity:
    """One entity as the dataset states it."""

    name: str
    type_name: str
    chunk_ids: frozenset[str]  # the dataset's chunks that mention it
