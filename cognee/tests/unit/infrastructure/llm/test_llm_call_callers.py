import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cognee.context_global_variables import current_caller, current_pipeline_stage
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.pipeline_stage import caller_scope, pipeline_stage
from cognee.infrastructure.session import feedback_detection
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.modules.improve.result import StageResult
from cognee.modules.improve.stage import BaseStage, execute_stage
from cognee.modules.search.types import SearchType

llm_utils = sys.modules["cognee.infrastructure.llm.utils"]
search_module = sys.modules["cognee.modules.search.methods.search"]


def _labels():
    return {"caller": current_caller.get(), "stage": current_pipeline_stage.get()}


def test_caller_scope_sets_only_the_caller():
    config = get_llm_context_config()

    with caller_scope("search_in_datasets_context"):
        assert _labels() == {"caller": "search_in_datasets_context", "stage": None}
        assert get_llm_context_config() is config

    assert _labels() == {"caller": None, "stage": None}


def test_pipeline_stage_leaves_the_caller_alone():
    with caller_scope("improve.distill_sessions"):
        with pipeline_stage("extraction"):
            assert _labels() == {"caller": "improve.distill_sessions", "stage": "extraction"}
        assert _labels() == {"caller": "improve.distill_sessions", "stage": None}


def test_the_nearest_caller_wins_and_the_outer_one_comes_back():
    with caller_scope("outer"):
        with caller_scope("inner"):
            assert current_caller.get() == "inner"
        assert current_caller.get() == "outer"


@pytest.mark.asyncio
async def test_an_improve_stage_is_the_caller_of_its_llm_calls():
    seen = {}

    class DistillStage(BaseStage):
        name = "distill_sessions"

        async def run(self, inputs):
            with pipeline_stage("extraction"):
                seen.update(_labels())
            return StageResult.skipped(self.name, "test")

    inputs = SimpleNamespace(config=SimpleNamespace(stages_disabled=[]), has_sessions=True)

    await execute_stage(DistillStage(), inputs)

    assert seen == {"caller": "improve.distill_sessions", "stage": "extraction"}
    assert _labels() == {"caller": None, "stage": None}


@pytest.mark.asyncio
async def test_a_search_is_the_caller_of_its_llm_calls(monkeypatch):
    seen = {}

    async def no_datasets(**kwargs):
        return []

    async def no_routing(*args, **kwargs):
        return None

    async def fake_search(**kwargs):
        with pipeline_stage("query"):
            seen.update(_labels())
        return []

    monkeypatch.setattr(search_module, "get_authorized_existing_datasets", no_datasets)
    monkeypatch.setattr(search_module, "_route_datasets", no_routing)
    monkeypatch.setattr(search_module, "search_in_datasets_context", fake_search)

    await search_module.authorized_search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="question", user=MagicMock()
    )

    assert seen == {"caller": "search_in_datasets_context", "stage": "query"}


@pytest.mark.asyncio
async def test_session_turn_analysis_is_the_caller_of_its_llm_call(monkeypatch):
    seen = {}

    async def fake_llm(**kwargs):
        seen.update(_labels())
        return SessionTurnAnalysis()

    monkeypatch.setattr(feedback_detection, "llm_available", lambda: True)
    monkeypatch.setattr(feedback_detection, "read_query_prompt", lambda name: "prompt")
    monkeypatch.setattr(feedback_detection, "is_auto_feedback_enabled", lambda: False)
    monkeypatch.setattr(feedback_detection.LLMGateway, "acreate_structured_output", fake_llm)

    with caller_scope("search_in_datasets_context"):
        await feedback_detection.analyze_turn_for_session_context("Thanks, that helped.")

    assert seen == {"caller": "analyze_turn_for_session_context", "stage": None}


@pytest.mark.asyncio
async def test_the_connection_check_is_the_caller_of_its_llm_call(monkeypatch):
    seen = {}

    async def fake_llm(**kwargs):
        seen.update(_labels())
        return "test"

    monkeypatch.setattr(llm_utils.LLMGateway, "acreate_structured_output", fake_llm)

    await llm_utils.test_llm_connection()

    assert seen == {"caller": "test_llm_connection", "stage": None}
