"""Temporal event for tracking skill lifecycle changes."""

from pydantic import Field

from cognee.modules.engine.models.Event import Event


class SkillChangeEvent(Event):
    """Records a skill add/update/remove for temporal queries.

    Extends Cognee's Event DataPoint so the TemporalRetriever can find it.
    """

    skill_id: str
    change_type: str = ""  # "added", "updated", "removed"
    old_content_hash: str = ""
    new_content_hash: str = ""
    skill_name: str = ""

    metadata: dict = Field(default_factory=lambda: {"index_fields": ["name", "description"]})
