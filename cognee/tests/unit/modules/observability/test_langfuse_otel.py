import os
from unittest.mock import patch, MagicMock

import pytest


def test_langfuse_config_enabled():
    fake_pk = "pk-12345"
    fake_sk = "sk-67890"
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": fake_pk,
            "LANGFUSE_SECRET_KEY": fake_sk,
            "LANGFUSE_HOST": "https://us.cloud.langfuse.com",
        },
        clear=True,
    ):
        from cognee.base_config import get_base_config

        get_base_config.cache_clear()
        config = get_base_config()

        assert (
            config.otel_exporter_otlp_endpoint
            == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
        )

        import base64

        expected_b64 = base64.b64encode(f"{fake_pk}:{fake_sk}".encode("utf-8")).decode("utf-8")
        assert config.otel_exporter_otlp_headers == f"Authorization=Basic {expected_b64}"
        assert config.cognee_tracing_enabled is True


def test_langfuse_config_missing_key():
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-123",
        },
        clear=True,
    ):
        from cognee.base_config import get_base_config

        get_base_config.cache_clear()

        with pytest.raises(ValueError, match="must be provided together"):
            get_base_config()


def test_langfuse_attribute_processor():
    from opentelemetry.sdk.trace import ReadableSpan
    from cognee.modules.observability.tracing import (
        LangfuseAttributeProcessor,
        COGNEE_LLM_MODEL,
        COGNEE_LLM_PROVIDER,
    )

    mock_exporter = MagicMock()
    processor = LangfuseAttributeProcessor(mock_exporter)

    span = MagicMock(spec=ReadableSpan)
    span._attributes = {
        COGNEE_LLM_MODEL: "gpt-4",
        COGNEE_LLM_PROVIDER: "openai",
        "cognee.llm.input_tokens": 10,
        "cognee.llm.cost": 0.01,
    }

    processor.on_end(span)

    assert span._attributes["gen_ai.request.model"] == "gpt-4"
    assert span._attributes["gen_ai.system"] == "openai"
    assert span._attributes["gen_ai.usage.input_tokens"] == 10
    assert span._attributes["gen_ai.usage.cost"] == 0.01


@patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter")
def test_langfuse_exporter_http_fallback(mock_exporter_cls):
    fake_pk = "pk-12345"
    fake_sk = "sk-67890"
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": fake_pk,
            "LANGFUSE_SECRET_KEY": fake_sk,
        },
        clear=True,
    ):
        from cognee.base_config import get_base_config
        from cognee.modules.observability.tracing import (
            _try_add_otlp_exporter,
            LangfuseAttributeProcessor,
        )

        get_base_config.cache_clear()

        mock_provider = MagicMock()
        _try_add_otlp_exporter(mock_provider)

        mock_exporter_cls.assert_called_once()
        kwargs = mock_exporter_cls.call_args.kwargs
        assert "https://cloud.langfuse.com/api/public/otel/v1/traces" in kwargs["endpoint"]
        assert "Authorization" in kwargs["headers"]

        import base64

        expected_b64 = base64.b64encode(f"{fake_pk}:{fake_sk}".encode("utf-8")).decode("utf-8")
        assert kwargs["headers"]["Authorization"] == f"Basic {expected_b64}"

        mock_provider.add_span_processor.assert_called_once()
        args = mock_provider.add_span_processor.call_args.args
        assert isinstance(args[0], LangfuseAttributeProcessor)
