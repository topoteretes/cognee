"""Loading the GLiNER demo model says so when it is about to download it (SDK-754).

The first keyless ``remember()`` fetches about 750 MB through the hub with no
word from cognee; the only trace was a progress bar on a terminal. The loader
now asks the local cache first and logs a warning naming the model, its size
and the cache location when a download is coming; a cached load stays at info.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cognee.tasks.graph.gliner_demo.extractor as extractor_module

MODEL = "fastino/gliner2.5-base-v1"


@pytest.fixture
def fake_gliner2(monkeypatch):
    """A stand-in ``gliner2`` package whose loader records the call."""
    from_pretrained = MagicMock(return_value=SimpleNamespace(name="extractor"))
    module = SimpleNamespace(AutoExtractor=SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "gliner2", module)
    monkeypatch.setattr(extractor_module, "_extractors", {})
    return from_pretrained


def test_first_use_download_is_announced(fake_gliner2, caplog):
    with (
        patch("huggingface_hub.try_to_load_from_cache", return_value=None),
        patch("huggingface_hub.constants.HF_HUB_CACHE", "/home/me/.cache/huggingface/hub"),
        caplog.at_level(logging.INFO, logger="gliner.extractor"),
    ):
        extractor_module.load_extractor(MODEL)

    fake_gliner2.assert_called_once_with(MODEL)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, [record.getMessage() for record in caplog.records]
    message = warnings[0].getMessage()
    assert "Downloading local model fastino/gliner2.5-base-v1 (about 750 MB)" in message
    assert "/home/me/.cache/huggingface/hub" in message
    assert "HF_HOME" in message and "once" in message
    assert any("ready in" in record.getMessage() for record in caplog.records)


def test_cached_model_loads_quietly(fake_gliner2, caplog):
    with (
        patch(
            "huggingface_hub.try_to_load_from_cache", return_value="/cache/snapshots/x/config.json"
        ),
        caplog.at_level(logging.INFO, logger="gliner.extractor"),
    ):
        extractor_module.load_extractor(MODEL)

    assert not [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Loading local model" in record.getMessage() for record in caplog.records)


def test_a_local_directory_counts_as_cached(tmp_path):
    cached, cache_dir = extractor_module.hub_model_cached(str(tmp_path))
    assert cached is True and cache_dir == str(tmp_path)


def test_unknown_model_reports_no_size(fake_gliner2, caplog):
    with (
        patch("huggingface_hub.try_to_load_from_cache", return_value=None),
        caplog.at_level(logging.WARNING, logger="gliner.extractor"),
    ):
        extractor_module.load_extractor("someone/other-gliner")

    assert "(size unknown)" in caplog.records[0].getMessage()
