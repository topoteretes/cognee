"""Persistence helpers for low-level add_data_points pipeline flow."""

from __future__ import annotations

from typing import Optional

import cognee
from cognee.low_level import DataPoint
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.storage import add_data_points


class AgentAction(DataPoint):
    """Agent action datapoint persisted through add/cognify text flow."""

    chain_root: "ActionChainRoot"
    iteration: int
    prev_action: Optional["AgentAction"] = None
    text: str
    metadata: dict = {"index_fields": ["text"]}


class ActionChainRoot(DataPoint):
    """Per-job root node for chaining all agent actions in the graph."""

    action_job_node: DataPoint
    job_id: str
    text: str
    metadata: dict = {"index_fields": ["text"]}


class JobSequenceNode(DataPoint):
    """Represents the ordinal position of a job in the demo run."""

    position: int
    previous: Optional["JobSequenceNode"] = None
    text: str
    metadata: dict = {"index_fields": ["text"]}


class ActionTaskJobNode(DataPoint):
    """Represents agent-task job grouping in the actions dataset."""

    position: int
    previous: Optional["ActionTaskJobNode"] = None
    text: str
    metadata: dict = {"index_fields": ["text"]}


class SkillStateSnapshot(DataPoint):
    """Historical snapshot of skill state connected to an action task job node."""

    action_job_node: ActionTaskJobNode
    job_id: str
    version: int
    feedbacks: list[str] = []
    previous_snapshot: Optional["SkillStateSnapshot"] = None
    text: str
    metadata: dict = {"index_fields": ["text"]}


def build_node_id_from_text(text: str):
    """Create deterministic UUID5 id from text."""
    return generate_node_id(text.strip())


def passthrough_data_points(data_points):
    """Pass datapoints through a custom pipeline task."""
    return data_points


async def persist_with_low_level_pipeline(
    data_points: list[DataPoint],
    dataset_name: str,
    user,
    pipeline_name: str,
) -> None:
    """Persist datapoints via low-level custom pipeline + add_data_points task."""
    if not data_points:
        return

    await cognee.run_custom_pipeline(
        tasks=[Task(passthrough_data_points), Task(add_data_points, context=None)],
        data=data_points,
        dataset=dataset_name,
        user=user,
        pipeline_name=pipeline_name,
    )
