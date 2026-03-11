"""Shared helpers for the skills package."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid5, UUID

from cognee.modules.engine.utils.generate_timestamp_datapoint import generate_timestamp_datapoint
from cognee.modules.engine.models.Timestamp import Timestamp

from cognee.cognee_skills.models.skill_change_event import SkillChangeEvent

EVENT_NAMESPACE = UUID("d4e5f6a7-b8c9-0123-def0-123456789abc")


def _now_timestamp() -> Timestamp:
    """Create a Cognee Timestamp DataPoint for the current UTC time."""
    now = datetime.now(timezone.utc)
    raw = Timestamp(
        id=UUID(int=0),
        time_at=0,
        year=now.year,
        month=now.month,
        day=now.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
        timestamp_str="",
    )
    return generate_timestamp_datapoint(raw)


def _make_change_event(
    skill_id: str,
    skill_name: str,
    change_type: str,
    old_hash: str = "",
    new_hash: str = "",
) -> SkillChangeEvent:
    ts = _now_timestamp()
    return SkillChangeEvent(
        id=uuid5(EVENT_NAMESPACE, f"{skill_id}:{change_type}:{ts.time_at}"),
        name=f"skill_{change_type}: {skill_name}",
        description=f"Skill '{skill_name}' ({skill_id}) was {change_type}.",
        skill_id=skill_id,
        change_type=change_type,
        old_content_hash=old_hash,
        new_content_hash=new_hash,
        skill_name=skill_name,
        at=ts,
    )
