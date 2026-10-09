from typing import Literal

from cognee.infrastructure.engine import DataPoint


class FactConflict(DataPoint):
    """A dataset's review of competing values for one Entity attribute."""

    dataset_id: str
    about_id: str
    attribute: str
    kind: Literal["time_varying", "fixed"]
    status: Literal["resolved", "unresolved"]
    text: str
    values: list[str] = []
    sources: list[str] = []
    # Remains true until links, marks, vectors and affected Entities are saved.
    review_pending: bool = False
    metadata: dict = {
        "index_fields": [],
        "identity_fields": ["dataset_id", "about_id", "attribute"],
    }
