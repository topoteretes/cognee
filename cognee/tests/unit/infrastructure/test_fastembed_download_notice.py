"""The fastembed engine says so when its model is about to be downloaded (SDK-754).

fastembed fetches the local embedder from the hub on first use with no word
from cognee. The engine now asks fastembed's cache first and logs a warning
naming the model, its size and the cache location when a download is coming;
a cached load stays at info.
"""

import logging
from unittest.mock import MagicMock, patch

import cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine as engine_module
from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
    fastembed_model_cached,
)

MODEL = "BAAI/bge-small-en-v1.5"
REGISTRY = [
    {
        "model": MODEL,
        "size_in_GB": 0.067,
        "sources": {"hf": "qdrant/bge-small-en-v1.5-onnx-q", "url": None},
    }
]


def test_cache_lookup_uses_fastembeds_own_location_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    with patch.object(engine_module.TextEmbedding, "list_supported_models", return_value=REGISTRY):
        cached, cache_dir, size_hint = fastembed_model_cached(MODEL)
        assert (cached, cache_dir, size_hint) == (False, str(tmp_path), "about 67 MB")

        # A hub download lands under models--<repo>; fastembed's own bucket under fast-<name>.
        (tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q").mkdir()
        assert fastembed_model_cached(MODEL)[0] is True
        (tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q").rmdir()
        (tmp_path / "fast-bge-small-en-v1.5").mkdir()
        assert fastembed_model_cached(MODEL)[0] is True


def test_unknown_model_has_no_size_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    with patch.object(engine_module.TextEmbedding, "list_supported_models", return_value=REGISTRY):
        cached, _, size_hint = fastembed_model_cached("someone/custom-embedder")
    assert (cached, size_hint) == (False, None)


def _engine(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    with (
        patch.object(engine_module, "TextEmbedding") as text_embedding,
        patch.object(FastembedEmbeddingEngine, "get_tokenizer", return_value=MagicMock()),
    ):
        text_embedding.list_supported_models.return_value = REGISTRY
        FastembedEmbeddingEngine(model=MODEL, dimensions=384)
        text_embedding.assert_called_once_with(model_name=MODEL)


def test_engine_announces_a_first_use_download(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="FastembedEmbeddingEngine"):
        _engine(monkeypatch, tmp_path)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, [record.getMessage() for record in caplog.records]
    message = warnings[0].getMessage()
    assert "Downloading local model BAAI/bge-small-en-v1.5 (about 67 MB)" in message
    assert str(tmp_path) in message and "FASTEMBED_CACHE_PATH" in message


def test_engine_loads_a_cached_model_quietly(tmp_path, monkeypatch, caplog):
    (tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q").mkdir()
    with caplog.at_level(logging.INFO, logger="FastembedEmbeddingEngine"):
        _engine(monkeypatch, tmp_path)

    assert not [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Loading local model" in record.getMessage() for record in caplog.records)
