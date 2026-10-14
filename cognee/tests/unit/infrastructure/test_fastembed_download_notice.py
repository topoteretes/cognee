"""The fastembed engine says so when its model is about to be downloaded (SDK-754).

fastembed fetches the local embedder from the hub on first use with no word
from cognee. The engine now asks fastembed's cache first and logs a warning
naming the model, its size and the cache location when a download is coming;
a cached load stays at info.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cognee.base_config as base_config_module
import cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine as engine_module
from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
    fastembed_model_cached,
    resolve_fastembed_cache_dir,
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


def _models_root(monkeypatch, path) -> None:
    """Serve a BaseConfig whose models root is ``path`` (COGNEE_MODELS_DIR is read
    at import time, so the field is set directly) and drop any FASTEMBED_CACHE_PATH."""
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    config = base_config_module.BaseConfig(models_root_directory=str(path))
    monkeypatch.setattr(base_config_module, "get_base_config", lambda: config)


def test_cache_defaults_under_cognee_models_dir(tmp_path, monkeypatch):
    _models_root(monkeypatch, tmp_path / "models")

    cache_dir = resolve_fastembed_cache_dir()

    assert Path(cache_dir).resolve() == (tmp_path / "models" / "fastembed").resolve()
    assert (tmp_path / "models" / "fastembed").is_dir()


def test_fastembed_cache_path_wins_over_models_dir(tmp_path, monkeypatch):
    _models_root(monkeypatch, tmp_path / "models")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "explicit"))

    # Taken as given: not created, not checked.
    assert resolve_fastembed_cache_dir() == str(tmp_path / "explicit")
    assert not (tmp_path / "models").exists()


def test_unusable_models_dir_falls_back_to_temp_with_a_warning(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "models"
    blocker.write_text("a file where the models directory should be")
    _models_root(monkeypatch, blocker)

    with caplog.at_level(logging.WARNING, logger="FastembedEmbeddingEngine"):
        cache_dir = resolve_fastembed_cache_dir()

    assert cache_dir == os.path.join(tempfile.gettempdir(), "fastembed_cache")
    warnings = [
        _event_text(record) for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1 and "COGNEE_MODELS_DIR" in warnings[0]


def test_unknown_model_has_no_size_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    with patch.object(engine_module.TextEmbedding, "list_supported_models", return_value=REGISTRY):
        cached, _, size_hint = fastembed_model_cached("someone/custom-embedder")
    assert (cached, size_hint) == (False, None)


def _event_text(record) -> str:
    """The log line itself, not its repr.

    cognee logs through structlog, so the record carries the event dict and
    ``getMessage()`` returns that dict's repr — in which a Windows path's
    backslashes come back escaped (``C:\\Users\\...``). Reading the event
    keeps these assertions identical on every platform.
    """
    return record.msg["event"] if isinstance(record.msg, dict) else record.getMessage()


def _engine(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    with (
        patch.object(engine_module, "TextEmbedding") as text_embedding,
        patch.object(FastembedEmbeddingEngine, "get_tokenizer", return_value=MagicMock()),
    ):
        text_embedding.list_supported_models.return_value = REGISTRY
        FastembedEmbeddingEngine(model=MODEL, dimensions=384)
        # The resolved directory is handed to fastembed explicitly; its own
        # default would be the system temp directory.
        text_embedding.assert_called_once_with(model_name=MODEL, cache_dir=str(tmp_path))


def test_engine_announces_a_first_use_download(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="FastembedEmbeddingEngine"):
        _engine(monkeypatch, tmp_path)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, [record.getMessage() for record in caplog.records]
    message = _event_text(warnings[0])
    assert "Downloading local model BAAI/bge-small-en-v1.5 (about 67 MB)" in message
    assert str(tmp_path) in message and "FASTEMBED_CACHE_PATH" in message


def test_engine_loads_a_cached_model_quietly(tmp_path, monkeypatch, caplog):
    (tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q").mkdir()
    with caplog.at_level(logging.INFO, logger="FastembedEmbeddingEngine"):
        _engine(monkeypatch, tmp_path)

    assert not [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Loading local model" in _event_text(record) for record in caplog.records)
