from typing import Literal

from cognee.infrastructure.engine import DataPoint


class FactConflict(DataPoint):
    """A conflict in one dataset: competing values for one Entity attribute."""

    dataset_id: str
    about_id: str
    attribute: str
    kind: Literal["time_varying", "fixed"]
    status: Literal["resolved", "unresolved"]
    text: str
    # An explicit user decision survives subsequent automatic conflict reviews.
    resolution_source: Literal["review", "user"] = "review"
    selected_fact_id: str | None = None
    selected_fact: str | None = None
    rejected_fact_ids: list[str] = []
    rejected_fact_texts: list[str] = []
    resolved_by: str | None = None
    resolved_at: str | None = None
    resolution_reason: str | None = None
    values: list[str] = []
    sources: list[str] = []
    # True from before a review run's first write of this conflict until that run's last write
    # has landed, so a run that stopped midway leaves it true for the next run to repair.
    write_pending: bool = False
    metadata: dict = {
        "index_fields": [],
        "identity_fields": ["dataset_id", "about_id", "attribute"],
    }
