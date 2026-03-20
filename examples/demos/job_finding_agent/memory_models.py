"""Cognee datapoints and persistence helpers for the demo."""

from __future__ import annotations

import cognee
from cognee.low_level import DataPoint
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.storage import add_data_points


class FormattedJob(DataPoint):
    """Structured job datapoint."""

    job_id: str
    role_title: str
    seniority: str
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: list[str] = []
    location_or_remote: str = ""
    raw_text: str
    metadata: dict = {"index_fields": ["job_id", "role_title", "raw_text"]}


class JobRecommendation(DataPoint):
    """Recommendation datapoint."""

    job_id: str
    decision: str
    rationale: str
    confidence: float
    metadata: dict = {"index_fields": ["job_id", "decision", "rationale"]}


class FeedbackRecord(DataPoint):
    """Feedback datapoint."""

    job_id: str
    decision: str
    feedback_text: str
    metadata: dict = {"index_fields": ["job_id", "feedback_text"]}


class AgentAction(DataPoint):
    """Agent step trace datapoint."""

    job_id: str
    iteration: int
    thought: str
    tool_name: str
    observation: str
    stop_reason: str = ""
    metadata: dict = {"index_fields": ["job_id", "tool_name", "observation"]}


def passthrough_data_points(data_points):
    """Pass datapoints through a custom pipeline task."""
    return data_points


async def persist_data_points(
    data_points: list[DataPoint],
    dataset_name: str,
    user,
    pipeline_name: str,
) -> None:
    """Persist datapoints using a tiny custom pipeline."""
    await cognee.run_custom_pipeline(
        tasks=[Task(passthrough_data_points), Task(add_data_points)],
        data=data_points,
        dataset=dataset_name,
        user=user,
        pipeline_name=pipeline_name,
    )

