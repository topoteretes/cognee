"""The improve stages that draft text with an LLM decline cleanly without one (SDK-753).

A keyless install (GLiNER + fastembed) can bridge sessions and traces into the
graph, but it cannot draft agent-context lessons, distil sessions or write the
global context summaries. Those stages used to run, build an LLM client that
raised, and end ``errored`` with a traceback in the log. Their gates now
report ``no_llm_configured`` instead, with zero LLM calls, like every other
gate. The trace-step summary follows the same rule: the deterministic
fallback, no client.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cognee.infrastructure.session.session_agent_trace as trace_module
import cognee.modules.improve.stages as stages_module
from cognee.modules.improve.stages import (
    REASON_NO_LLM_CONFIGURED,
    REASON_OPT_IN_DISABLED,
    DistillSessionsStage,
    ExtractAgentContextStage,
    GlobalContextIndexStage,
)


def _inputs(**overrides):
    return SimpleNamespace(**{"build_global_context_index": True, **overrides})


def _session_manager(available=True, auto_feedback=True):
    manager = MagicMock()
    manager.is_available = available
    manager.is_auto_feedback_enabled.return_value = auto_feedback
    return manager


@pytest.mark.parametrize(
    "stage",
    [ExtractAgentContextStage(), DistillSessionsStage(), GlobalContextIndexStage()],
    ids=lambda stage: stage.name,
)
def test_llm_stages_decline_without_a_usable_llm(stage):
    with (
        patch.object(stages_module, "llm_available", return_value=False),
        patch(
            "cognee.infrastructure.session.get_session_manager.get_session_manager",
            return_value=_session_manager(),
        ),
    ):
        assert stage.gate(_inputs()) == REASON_NO_LLM_CONFIGURED


@pytest.mark.parametrize(
    "stage",
    [ExtractAgentContextStage(), DistillSessionsStage(), GlobalContextIndexStage()],
    ids=lambda stage: stage.name,
)
def test_llm_stages_run_when_an_llm_is_configured(stage):
    with (
        patch.object(stages_module, "llm_available", return_value=True),
        patch(
            "cognee.infrastructure.session.get_session_manager.get_session_manager",
            return_value=_session_manager(),
        ),
    ):
        assert stage.gate(_inputs()) is None


def test_earlier_gates_keep_precedence_over_the_llm_check():
    with patch.object(stages_module, "llm_available", return_value=False):
        assert (
            GlobalContextIndexStage().gate(_inputs(build_global_context_index=False))
            == REASON_OPT_IN_DISABLED
        )


@pytest.mark.asyncio
async def test_trace_summary_falls_back_without_building_an_llm_client():
    with (
        patch.object(trace_module, "llm_available", return_value=False),
        patch.object(
            trace_module.LLMGateway, "acreate_structured_output", new_callable=AsyncMock
        ) as llm,
    ):
        feedback = await trace_module.generate_agent_trace_feedback(
            origin_function="search_docs",
            status="success",
            method_return_value={"hits": 3},
        )

    assert feedback == trace_module.fallback_agent_trace_feedback(
        origin_function="search_docs", status="success", error_message=""
    )
    llm.assert_not_awaited()
