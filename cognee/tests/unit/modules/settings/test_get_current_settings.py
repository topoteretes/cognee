"""The telemetry provider-stack payload (SDK-775): reports the embedder and the
extractor as they would resolve now, and can never raise — it is computed
before a pipeline's first event."""

import importlib

import cognee.modules.preflight as preflight_module

# The package re-exports the function under the module's name; fetch the module itself.
settings_module = importlib.import_module("cognee.modules.settings.get_current_settings")


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


def test_keyless_install_reports_the_demo_extractor_without_touching_the_runtime(monkeypatch):
    """The GLiNER runtime is installed (or refused) by ``ensure_extractor_runtime``
    at cognify time. The telemetry payload only reports the decision."""
    monkeypatch.setattr(preflight_module, "keyless_local_defaults_apply", lambda *_: True)

    payload = settings_module.get_current_settings()

    assert payload["graph_extractor"] == "gliner_demo"


def test_model_settings_that_are_paths_leave_as_the_local_path_label(monkeypatch):
    """A model pointing at a local file names the OS account; telemetry carries a label."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        settings_module,
        "get_llm_config",
        lambda: SimpleNamespace(llm_provider="custom", llm_model="/Users/alice/models/x.gguf"),
    )
    monkeypatch.setattr(
        settings_module, "resolve_embedding_names", lambda *_: ("custom", "/home/alice/embed")
    )

    payload = settings_module.get_current_settings()

    assert payload["llm"] == {"provider": "custom", "model": "local_path"}
    assert payload["embedding"] == {"provider": "custom", "model": "local_path"}
