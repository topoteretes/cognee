"""Filesystem-like model identifiers never leave the process in telemetry.

Local GGUF / directory setups put a path (often with an OS account name) in
``llm.model`` or ``embedding.model``. ``send_telemetry`` rewrites those to a
basename before send; hosted names pass through.
"""

import asyncio
from typing import Any

import pytest

from cognee.shared import utils


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("/home/alice/models/llama.gguf", "llama.gguf"),
        ("openai//home/alice/models/llama.gguf", "openai/llama.gguf"),
        (r"C:\Users\alice\model.gguf", "model.gguf"),
        ("openai/gpt-4o", "openai/gpt-4o"),
        ("/", "local-path"),
        ("openai//", "openai/local-path"),
        ("C:\\", "local-path"),
        ("~/models/llama.gguf", "llama.gguf"),
        (r"custom/C:\Users\alice\model.gguf", "custom/model.gguf"),
    ],
)
def test_rewrite_path_like_model(model, expected):
    assert utils._rewrite_path_like_model(model) == expected


def test_rewrite_nested_model_paths_covers_llm_and_embedding():
    out = utils._rewrite_nested_model_paths(
        {
            "llm": {"provider": "openai", "model": "/home/alice/models/llama.gguf"},
            "embedding": {"provider": "custom", "model": r"C:\Users\alice\model.gguf"},
            "other": {"model": "openai/gpt-4o"},
        }
    )

    assert out["llm"]["model"] == "llama.gguf"
    assert out["embedding"]["model"] == "model.gguf"
    assert out["other"]["model"] == "openai/gpt-4o"
    assert out["llm"]["provider"] == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("properties", "expected_llm", "expected_embedding"),
    [
        (
            {
                "llm": {"model": "/home/alice/models/llama.gguf"},
                "embedding": {"model": "openai//home/alice/models/llama.gguf"},
            },
            "llama.gguf",
            "openai/llama.gguf",
        ),
        (
            {
                "llm": {"model": r"C:\Users\alice\model.gguf"},
                "embedding": {"model": "/"},
            },
            "model.gguf",
            "local-path",
        ),
        (
            {
                "llm": {"model": "openai/gpt-4o"},
                "embedding": {"model": "text-embedding-3-small"},
            },
            "openai/gpt-4o",
            "text-embedding-3-small",
        ),
    ],
)
async def test_send_telemetry_rewrites_nested_model_paths(
    monkeypatch, properties, expected_llm, expected_embedding
):
    payloads: list[dict[str, Any]] = []

    async def capture(payload: dict[str, Any]) -> None:
        payloads.append(payload)

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "_send_telemetry_request", capture)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anon")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent")

    utils.send_telemetry("Pipeline Run Started", None, additional_properties=properties)
    await asyncio.sleep(0)

    assert len(payloads) == 1
    sent = payloads[0]["properties"]
    assert sent["llm"]["model"] == expected_llm
    assert sent["embedding"]["model"] == expected_embedding
    assert "/home/" not in repr(payloads[0])
    assert "/Users/" not in repr(payloads[0])
    assert "alice" not in repr(payloads[0])
