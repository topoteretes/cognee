from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
    persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
)


@pytest.mark.asyncio
async def test_persist_agent_trace_feedbacks_pipeline_wires_memify_tasks():
    user = MagicMock()
    user.id = "u1"
    authorized_dataset = SimpleNamespace(id="dataset-1", owner_id="owner-1")

    with (
        patch(
            "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.set_session_user_context_variable",
            new=AsyncMock(),
        ) as set_user_ctx,
        patch(
            "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[authorized_dataset]),
        ) as get_authorized_dataset,
        patch(
            "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.set_database_global_context_variables",
            new=AsyncMock(),
        ) as set_db_ctx,
        patch(
            "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.memify",
            new=AsyncMock(return_value={"status": "ok"}),
        ) as memify_mock,
    ):
        result = await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=["s1", "s2"],
            dataset="main_dataset",
            node_set_name="custom_feedbacks",
            raw_trace_content=True,
            last_n_steps=3,
        )

    assert result == {"status": "ok"}
    set_user_ctx.assert_awaited_once_with(user)
    get_authorized_dataset.assert_awaited_once()
    set_db_ctx.assert_awaited_once_with("dataset-1", "owner-1")

    memify_kwargs = memify_mock.call_args.kwargs
    assert memify_kwargs["dataset"] == "dataset-1"
    assert memify_kwargs["data"] == [{}]
    assert len(memify_kwargs["extraction_tasks"]) == 1
    assert len(memify_kwargs["enrichment_tasks"]) == 1

    extraction_task = memify_kwargs["extraction_tasks"][0]
    enrichment_task = memify_kwargs["enrichment_tasks"][0]
    assert extraction_task.default_params["kwargs"]["session_ids"] == ["s1", "s2"]
    assert extraction_task.default_params["kwargs"]["raw_trace_content"] is True
    assert extraction_task.default_params["kwargs"]["last_n_steps"] == 3
    assert enrichment_task.default_params["kwargs"]["dataset_id"] == "dataset-1"
    assert enrichment_task.default_params["kwargs"]["node_set_name"] == "custom_feedbacks"
