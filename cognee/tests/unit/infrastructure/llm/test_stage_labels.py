import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cognee.context_global_variables import current_pipeline_stage
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.session import feedback_detection
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.modules.improve.result import StageResult
from cognee.modules.improve.stage import BaseStage, execute_stage
from cognee.modules.search.types import SearchType

llm_utils = sys.modules["cognee.infrastructure.llm.utils"]
search_module = sys.modules["cognee.modules.search.methods.search"]


def test_a_label_only_name_leaves_the_model_config_unchanged():
    before = get_llm_context_config()

    with pipeline_stage("retrieval"):
        assert current_pipeline_stage.get() == "retrieval"
        assert get_llm_context_config() is before

    assert current_pipeline_stage.get() is None


def test_an_inner_stage_wins_and_the_outer_label_comes_back():
    with pipeline_stage("retrieval"):
        with pipeline_stage("query"):
            assert current_pipeline_stage.get() == "query"
        assert current_pipeline_stage.get() == "retrieval"


@pytest.mark.asyncio
async def test_an_improve_stage_labels_its_llm_calls_with_its_name():
    seen = {}

    class ReviewStage(BaseStage):
        name = "review_conflicts"

        async def run(self, inputs):
            seen["stage"] = current_pipeline_stage.get()
            return StageResult.skipped(self.name, "test")

    inputs = SimpleNamespace(config=SimpleNamespace(stages_disabled=[]), has_sessions=True)

    await execute_stage(ReviewStage(), inputs)

    assert seen["stage"] == "improve.review_conflicts"
    assert current_pipeline_stage.get() is None


@pytest.mark.asyncio
async def test_a_search_labels_its_llm_calls_as_retrieval(monkeypatch):
    seen = {}

    async def no_datasets(**kwargs):
        return []

    async def no_routing(*args, **kwargs):
        return None

    async def fake_search(**kwargs):
        seen["stage"] = current_pipeline_stage.get()
        return []

    monkeypatch.setattr(search_module, "get_authorized_existing_datasets", no_datasets)
    monkeypatch.setattr(search_module, "_route_datasets", no_routing)
    monkeypatch.setattr(search_module, "search_in_datasets_context", fake_search)

    await search_module.authorized_search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="question", user=MagicMock()
    )

    assert seen["stage"] == "retrieval"


@pytest.mark.asyncio
async def test_session_turn_analysis_labels_its_llm_call(monkeypatch):
    seen = {}

    async def fake_llm(**kwargs):
        seen["stage"] = current_pipeline_stage.get()
        return SessionTurnAnalysis()

    monkeypatch.setattr(feedback_detection, "llm_available", lambda: True)
    monkeypatch.setattr(feedback_detection, "read_query_prompt", lambda name: "prompt")
    monkeypatch.setattr(feedback_detection, "is_auto_feedback_enabled", lambda: False)
    monkeypatch.setattr(feedback_detection.LLMGateway, "acreate_structured_output", fake_llm)

    with pipeline_stage("retrieval"):
        await feedback_detection.analyze_turn_for_session_context("Thanks, that helped.")

    assert seen["stage"] == "session_turn_analysis"


@pytest.mark.asyncio
async def test_the_connection_check_labels_its_llm_call(monkeypatch):
    seen = {}

    async def fake_llm(**kwargs):
        seen["stage"] = current_pipeline_stage.get()
        return "test"

    monkeypatch.setattr(llm_utils.LLMGateway, "acreate_structured_output", fake_llm)

    await llm_utils.test_llm_connection()

    assert seen["stage"] == "connection_test"
