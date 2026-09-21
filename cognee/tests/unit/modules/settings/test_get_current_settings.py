"""The telemetry provider-stack payload (SDK-775): reports the embedder and the
extractor as they would resolve now, and can never raise — it is computed
before a pipeline's first event."""

import importlib
import importlib.util

import pytest

import cognee.modules.preflight as preflight_module
from cognee.modules.cognify import config as cognify_config

# The package re-exports the function under the module's name; fetch the module itself.
settings_module = importlib.import_module("cognee.modules.settings.get_current_settings")

_real_find_spec = importlib.util.find_spec


def _without_gliner2(name, *args, **kwargs):
    return None if name == "gliner2" else _real_find_spec(name, *args, **kwargs)


def test_payload_reports_embedder_and_extractor(monkeypatch):
    monkeypatch.setattr(
        settings_module,
        "resolve_embedding_names",
        lambda *_: ("fastembed", "BAAI/bge-small-en-v1.5"),
    )
    monkeypatch.setattr(settings_module, "resolve_extractor_name", lambda *_: "gliner_demo")

    payload = settings_module.get_current_settings()

    assert set(payload) == {"llm", "embedding", "graph_extractor", "graph", "vector", "relational"}
    assert payload["embedding"] == {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"}
    assert payload["graph_extractor"] == "gliner_demo"


def test_unknown_extractor_setting_is_never_echoed(monkeypatch):
    monkeypatch.setattr(settings_module, "resolve_extractor_name", lambda *_: "my-private-fork")

    assert settings_module.get_current_settings()["graph_extractor"] == "invalid"


def test_keyless_install_without_gliner2_does_not_raise(monkeypatch):
    """cognify raises KeylessExtractorNotInstalledError for this state — at
    cognify time. The telemetry payload must report the extractor instead."""
    monkeypatch.setattr(preflight_module, "keyless_local_defaults_apply", lambda *_: True)
    monkeypatch.setattr(cognify_config.importlib.util, "find_spec", _without_gliner2)

    payload = settings_module.get_current_settings()

    assert payload["graph_extractor"] == "gliner_demo"
    with pytest.raises(cognify_config.KeylessExtractorNotInstalledError):
        cognify_config.resolve_extractor(None, cognify_config.CognifyConfig(graph_extractor="auto"))
