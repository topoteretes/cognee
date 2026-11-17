import importlib
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


def _recording_llm(seen, result):
    async def fake_llm(*args, **kwargs):
        seen.update(_labels())
        return result

    return fake_llm


@pytest.mark.asyncio
async def test_agent_trace_feedback_is_the_caller_of_its_llm_call(monkeypatch):
    from cognee.infrastructure.session import session_agent_trace

    seen = {}
    monkeypatch.setattr(session_agent_trace, "read_query_prompt", lambda name: "prompt")
    monkeypatch.setattr(
        session_agent_trace.LLMGateway,
        "acreate_structured_output",
        _recording_llm(seen, SimpleNamespace(session_feedback="done")),
    )

    await session_agent_trace.generate_agent_trace_feedback(
        origin_function="tool", status="success", method_return_value={"ok": True}
    )

    assert seen == {"caller": "generate_agent_trace_feedback", "stage": None}


@pytest.mark.asyncio
async def test_trace_batch_extraction_is_the_caller_of_its_llm_call(monkeypatch):
    from cognee.infrastructure.session import agent_context_extraction

    async def no_lessons(*args, **kwargs):
        return []

    seen = {}
    monkeypatch.setattr(agent_context_extraction, "build_trace_batch", lambda traces: "batch")
    monkeypatch.setattr(agent_context_extraction, "read_query_prompt", lambda name: "prompt")
    monkeypatch.setattr(agent_context_extraction, "_existing_agent_lessons", no_lessons)
    monkeypatch.setattr(
        agent_context_extraction.LLMGateway,
        "acreate_structured_output",
        _recording_llm(seen, SimpleNamespace(lessons=[])),
    )

    await agent_context_extraction._extract_batch_from_traces(
        session_manager=MagicMock(), user_id="u", session_id="s", traces=[object()]
    )

    assert seen == {"caller": "_extract_batch_from_traces", "stage": None}


@pytest.mark.asyncio
async def test_pii_assessment_is_the_caller_of_its_llm_call(monkeypatch):
    from cognee.tasks.presort.models import FileRecord

    detect_pii = importlib.import_module("cognee.tasks.presort.detect_pii")

    seen = {}
    monkeypatch.setattr(detect_pii, "render_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(
        detect_pii.LLMGateway,
        "acreate_structured_output",
        _recording_llm(seen, SimpleNamespace(contains_personal_data=False)),
    )

    await detect_pii._llm_assess(FileRecord(path="/notes.txt", name="notes.txt"), "text")

    assert seen == {"caller": "detect_pii._llm_assess", "stage": None}


def test_span_enrichment_writes_the_caller_and_the_stage(monkeypatch):
    otel_trace = pytest.importorskip("opentelemetry.trace")
    from cognee.infrastructure.llm.structured_output_framework.litellm_native import (
        native_adapter,
    )
    from cognee.modules.observability import trace_context

    class Span:
        def __init__(self):
            self.attributes = {}

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = Span()
    monkeypatch.setattr(trace_context, "is_tracing_enabled", lambda: True)
    monkeypatch.setattr(otel_trace, "get_current_span", lambda: span)

    with caller_scope("search_in_datasets_context"), pipeline_stage("query"):
        native_adapter._enrich_llm_span("openai/gpt-5-mini", "LiteLLM-Native")

    assert span.attributes["cognee.llm.caller"] == "search_in_datasets_context"
    assert span.attributes["cognee.pipeline.stage"] == "query"
    assert span.attributes["cognee.llm.model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_the_native_plain_text_and_fallback_paths_enrich_their_span(monkeypatch):
    from litellm.exceptions import ContentPolicyViolationError

    from cognee.infrastructure.llm.structured_output_framework.litellm_native import (
        native_adapter,
    )

    enriched = []
    monkeypatch.setattr(
        native_adapter, "_enrich_llm_span", lambda model, name: enriched.append(model)
    )
    adapter = native_adapter.NativeLiteLLMAdapter(
        api_key="key",
        model="primary",
        max_completion_tokens=100,
        fallback_model="fallback",
        fallback_api_key="fallback-key",
    )

    async def plain_text(*args, **kwargs):
        return "answer"

    async def structured(text_input, system_prompt, response_model, *, model, **kwargs):
        if model == "primary":
            raise ContentPolicyViolationError("blocked", model="primary", llm_provider="openai")
        return SessionTurnAnalysis()

    monkeypatch.setattr(adapter, "_acreate_str_output", plain_text)
    monkeypatch.setattr(adapter, "_acreate_structured", structured)

    await adapter.acreate_structured_output("text", "prompt", str)
    await adapter.acreate_structured_output("text", "prompt", SessionTurnAnalysis)

    assert enriched == ["primary", "fallback"]
