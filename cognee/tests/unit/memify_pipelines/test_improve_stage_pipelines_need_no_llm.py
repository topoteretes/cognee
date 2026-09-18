"""The improve-stage pipelines declare that none of their tasks needs an LLM.

``Task`` defaults to ``needs_llm=True`` and the pipeline setup layer runs the
LLM connection probe whenever any task claims to need one. The three
session-fed improve stages (feedback weights, session Q&A bridging, agent
traces) make no LLM call of their own — the bridging tasks run ``cognify()``,
which resolves its own extractor — so with the default left in place a
keyless install (GLiNER + fastembed) failed the bridge on the probe, before
any task ran, and session content never reached the graph (SDK-753).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline
from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
    persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
)
from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.modules.pipelines.tasks.task import pipeline_needs_llm

PIPELINES = [
    ("cognee.memify_pipelines.apply_feedback_weights", apply_feedback_weights_pipeline),
    (
        "cognee.memify_pipelines.persist_sessions_in_knowledge_graph",
        persist_sessions_in_knowledge_graph_pipeline,
    ),
    (
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph",
        persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "pipeline"), PIPELINES, ids=[m.rsplit(".", 1)[1] for m, _ in PIPELINES]
)
async def test_stage_pipeline_tasks_need_no_llm(module, pipeline):
    user = MagicMock()
    user.id = uuid4()
    dataset = SimpleNamespace(id=uuid4(), owner_id=user.id)

    with (
        patch(f"{module}.set_session_user_context_variable", new=AsyncMock()),
        patch(
            f"{module}.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch(f"{module}.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock,
    ):
        await pipeline(user=user, session_ids=["s1"])

    kwargs = memify_mock.call_args.kwargs
    tasks = kwargs["extraction_tasks"] + kwargs["enrichment_tasks"]
    assert tasks, "the pipeline must hand memify its tasks"
    assert all(task.needs_llm is False for task in tasks), [
        (task.executable.__name__, task.needs_llm) for task in tasks
    ]
    assert pipeline_needs_llm(tasks) is False, "the setup layer would run the LLM probe"
