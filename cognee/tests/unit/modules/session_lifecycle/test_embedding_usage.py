"""Embedding usage belongs in the same totals as server-side generation."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.modules.operations.usage_accumulator import operation_usage_scope
from cognee.modules.session_lifecycle import metrics
from cognee.modules.session_lifecycle import usage_tracking as tracking
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord


@pytest.mark.asyncio
async def test_embedding_usage_preserves_provider_tokens_and_cost(monkeypatch):
    record = AsyncMock()
    monkeypatch.setattr(tracking, "record_llm_call", record)
    monkeypatch.setattr("litellm.completion_cost", lambda **kw: 0.001)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=37))
    await tracking.record_embedding_call(response, model="embedding-model", input_text="text")
    assert record.call_args.kwargs == {
        "input_text": "text",
        "output_text": "",
        "model": "embedding-model",
        "tokens_in_override": 37,
        "tokens_out_override": 0,
        "cost_usd_override": 0.001,
        "count_in_operation": False,
        "update_last_model": False,
    }


@pytest.mark.asyncio
async def test_unpriced_embedding_retains_tokens_and_cannot_break_capture(monkeypatch):
    record = AsyncMock(side_effect=OSError("metrics unavailable"))
    monkeypatch.setattr(tracking, "record_llm_call", record)

    def unknown_price(**kw):
        raise ValueError("unknown model")

    monkeypatch.setattr("litellm.completion_cost", unknown_price)
    await tracking.record_embedding_call(
        SimpleNamespace(usage={"prompt_tokens": 12}), model="custom", input_text="text"
    )
    assert record.call_args.kwargs["tokens_in_override"] == 12
    assert record.call_args.kwargs["cost_usd_override"] is None


@pytest.mark.asyncio
async def test_embedding_tokens_stay_out_of_operation_counters(monkeypatch):
    """Operation rows count LLM tokens only; an embedding batch must not move them."""
    monkeypatch.setattr("litellm.completion_cost", lambda **kw: 0.0)

    with operation_usage_scope() as usage:
        await tracking.record_llm_call(
            input_text="", output_text="", tokens_in_override=2500, tokens_out_override=400
        )
        await tracking.record_embedding_call(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10_000)),
            model="openai/text-embedding-3-large",
            input_text="text",
        )

    assert (usage.tokens_in, usage.tokens_out) == (2500, 400)


@pytest.mark.asyncio
async def test_embedding_keeps_the_answering_model_as_last_model(monkeypatch):
    """The embedder gets its own per-model row but never becomes last_model."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_path}")
    try:
        await engine.create_database()
        monkeypatch.setattr(metrics, "get_relational_engine", lambda: engine)
        monkeypatch.setattr("litellm.completion_cost", lambda **kw: 0.0002)

        session_id, user_id = "trial-1", uuid4()
        await metrics.ensure_and_touch_session(session_id=session_id, user_id=user_id)

        async with tracking.track_session_usage(session_id, user_id):
            await tracking.record_llm_call(
                input_text="",
                output_text="",
                model="openai/gpt-5-mini",
                tokens_in_override=2500,
                tokens_out_override=400,
            )
            await tracking.record_embedding_call(
                SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10_000)),
                model="openai/text-embedding-3-small",
                input_text="text",
            )

        async with engine.get_async_session() as session:
            record = (
                await session.execute(
                    select(SessionRecord).where(SessionRecord.session_id == session_id)
                )
            ).scalar_one()
            per_model = {
                row.model: row.tokens_in
                for row in (await session.execute(select(SessionModelUsage))).scalars()
            }

        assert record.last_model == "openai/gpt-5-mini"
        assert record.tokens_in == 12_500
        assert per_model == {
            "openai/gpt-5-mini": 2500,
            "openai/text-embedding-3-small": 10_000,
        }
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


def _embedding_engine(monkeypatch, aembedding):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", aembedding)
    return LiteLLMEmbeddingEngine(dimensions=4)


def _embedding_response(prompt_tokens: int):
    return SimpleNamespace(
        data=[{"embedding": [1.0, 1.0, 1.0, 1.0]}],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
    )


@pytest.mark.asyncio
async def test_embed_text_records_the_provider_response(monkeypatch):
    response = _embedding_response(7)
    engine = _embedding_engine(monkeypatch, AsyncMock(return_value=response))
    record = AsyncMock()
    monkeypatch.setattr(tracking, "record_embedding_call", record)

    vectors = await engine.embed_text(["hello"])

    assert vectors == [[1.0, 1.0, 1.0, 1.0]]
    record.assert_awaited_once()
    assert record.await_args.args == (response,)
    assert record.await_args.kwargs == {"model": engine.model, "input_text": "hello"}


@pytest.mark.asyncio
async def test_instrumentation_failure_never_resends_the_paid_request(monkeypatch):
    """A failing usage write must not trip embed_text's retry and bill twice."""
    aembedding = AsyncMock(return_value=_embedding_response(7))
    engine = _embedding_engine(monkeypatch, aembedding)
    monkeypatch.setattr(tracking, "record_llm_call", AsyncMock(side_effect=RuntimeError("db down")))

    vectors = await engine.embed_text(["hello"])

    assert vectors == [[1.0, 1.0, 1.0, 1.0]]
    assert aembedding.await_count == 1
