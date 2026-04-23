"""Temporal event for tracking skill lifecycle changes."""

from typing import Optional
from cognee.modules.engine.models.Event import Event
from cognee.modules.engine.models.Timestamp import Timestamp


class SkillChangeEvent(Event):
    """Records a skill add/update/remove for temporal queries.

    Extends Cognee's Event DataPoint so the TemporalRetriever can find it.
    """

    skill_id: str
    change_type: str = ""  # "added", "updated", "removed"
    old_content_hash: str = ""
    new_content_hash: str = ""
    skill_name: str = ""

    metadata: dict = {"index_fields": ["name", "description"]}
