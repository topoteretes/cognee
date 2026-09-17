"""Data models for the user-preference store."""

from cognee.infrastructure.engine import InternalDataPoint
from cognee.infrastructure.engine.models.DataPoint import MetaData


class UserPreference(InternalDataPoint):
    """At most one node per (user, dataset): what that user cares about.

    ``text`` holds preference lines only — the render header is added at
    render time. ``turn_counter`` is the only clock for weight decay; it
    advances once per processed turn, rated or not. ``text_watermark`` is the
    ``created_at`` of the newest context entry folded into ``text``, so a
    re-run never folds the same entry twice.

    Extends ``InternalDataPoint``: never embedded (empty ``index_fields``),
    never surfaced in retrieval output (``is_internal`` read at the graph
    read chokepoints). It intentionally carries no ``name``.
    """

    user_id: str
    dataset_id: str
    text: str = ""
    turn_counter: int = 0
    text_watermark: str = ""
    metadata: MetaData = {"index_fields": []}
