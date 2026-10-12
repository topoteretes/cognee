"""Regression test for #5144: embedding calls emit a standard GenAI span."""

import pytest


@pytest.mark.asyncio
async def test_embed_text_emits_genai_embeddings_span(monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    try:
        import opentelemetry
    except ImportError:
        pytest.skip("opentelemetry-sdk not installed")

    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )
    from cognee.modules.observability.trace_context import (
        clear_traces,
        disable_tracing,
        enable_tracing,
    )
    from cognee.modules.observability.tracing import get_exporter

    enable_tracing()
    try:
        clear_traces()
        engine = LiteLLMEmbeddingEngine(
            model="openai/text-embedding-3-large",
            provider="openai",
            dimensions=32,
        )
        vecs = await engine.embed_text(["hello world", "second doc"])
        assert len(vecs) == 2

        exporter = get_exporter()
        assert exporter is not None
        all_spans = [s for spans in exporter.get_all_traces().values() for s in spans]
        embedding_spans = [
            s for s in all_spans if s["attributes"].get("gen_ai.operation.name") == "embeddings"
        ]
        assert len(embedding_spans) == 1
        attrs = embedding_spans[0]["attributes"]
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.request.model"] == "openai/text-embedding-3-large"
        assert attrs["gen_ai.embeddings.dimension.count"] == 32
    finally:
        disable_tracing()
