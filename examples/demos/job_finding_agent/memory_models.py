"""Cognee datapoints and persistence helpers for the demo."""

from __future__ import annotations

from typing import Any

import cognee
from cognee.low_level import DataPoint
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.modules.pipelines.tasks.task import Task
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges


class FormattedJob(DataPoint):
    """Structured job datapoint."""

    job_id: str
    role_title: str = ""
    seniority: str = ""
    text: str
    metadata: dict = {"index_fields": ["text"]}


class JobRecommendation(DataPoint):
    """Recommendation datapoint."""

    job_id: str
    decision: str
    confidence: float
    text: str
    metadata: dict = {"index_fields": ["text"]}


class FeedbackRecord(DataPoint):
    """Feedback datapoint."""

    job_id: str
    text: str
    metadata: dict = {"index_fields": ["text"]}


class AgentAction(DataPoint):
    """Agent step trace datapoint."""

    job_id: str
    iteration: int
    thought: str
    tool_name: str
    text: str
    metadata: dict = {"index_fields": ["text"]}


def build_node_id_from_text(text: str):
    """Create deterministic UUID5 id from text."""
    return generate_node_id(text.strip())


def passthrough_data_points(data_points):
    """Pass datapoints through a custom pipeline task."""
    return data_points


async def persist_nodes_only(data_points: list[DataPoint]):
    """Persist datapoints as graph nodes and vector entries without edge writes."""
    if not isinstance(data_points, list):
        return data_points
    if not data_points:
        return data_points

    unified = await get_unified_engine()
    await unified.graph.add_nodes(data_points)
    await index_data_points(data_points, vector_engine=unified.vector)
    return data_points


async def persist_edges_only(edges: list[tuple[str, str, str, dict[str, Any]]]):
    """Persist graph edges with default edge properties and index them."""
    if not isinstance(edges, list):
        return edges
    if not edges:
        return edges

    normalized_edges = ensure_default_edge_properties(edges)
    unified = await get_unified_engine()
    await unified.graph.add_edges(normalized_edges)
    await index_graph_edges(normalized_edges, vector_engine=unified.vector)
    return normalized_edges


async def persist_data_points(
    data_points: list[DataPoint],
    edges: list[tuple[str, str, str, dict[str, Any]]] | None,
    dataset_name: str,
    user,
    pipeline_name: str,
) -> None:
    """Persist datapoints using a tiny custom pipeline."""
    await cognee.run_custom_pipeline(
        tasks=[
            Task(passthrough_data_points),
            Task(persist_nodes_only),
        ],
        data=data_points or [],
        dataset=dataset_name,
        user=user,
        pipeline_name=pipeline_name,
    )

    if edges:
        await cognee.run_custom_pipeline(
            tasks=[Task(persist_edges_only)],
            data=edges,
            dataset=dataset_name,
            user=user,
            pipeline_name=f"{pipeline_name}_edges",
        )
