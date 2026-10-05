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


def test_llm_block_says_whether_the_llm_is_usable(monkeypatch):
    """A keyless install reports the default llm provider/model; ``configured``
    tells a usable LLM from an unused default."""
    monkeypatch.setattr(settings_module, "llm_available", lambda *_: False)
    assert settings_module.get_current_settings()["llm"]["configured"] is False

    monkeypatch.setattr(settings_module, "llm_available", lambda *_: True)
    assert settings_module.get_current_settings()["llm"]["configured"] is True


def test_model_settings_that_are_paths_leave_as_the_local_path_label(monkeypatch):
    """A model pointing at a local file names the OS account; telemetry carries a label."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        settings_module,
        "get_llm_context_config",
        lambda: SimpleNamespace(
            llm_provider="custom", llm_model="/Users/alice/models/x.gguf", llm_api_key=""
        ),
    )
    monkeypatch.setattr(
        settings_module, "resolve_embedding_names", lambda *_: ("custom", "/home/alice/embed")
    )

    payload = settings_module.get_current_settings()

    assert payload["llm"] == {
        "provider": "custom",
        "model": "local_path",
        "configured": False,
        "structured_output": "unknown",
        "instructor_mode": "unknown",
    }
    assert payload["embedding"] == {"provider": "custom", "model": "local_path"}


def test_llm_half_reads_the_same_context_config_as_the_embedding_half(monkeypatch):
    """A per-call LLMConfig must describe the whole event, not only its embedder."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        settings_module,
        "get_llm_context_config",
        lambda: SimpleNamespace(
            llm_provider="anthropic", llm_model="anthropic/claude", llm_api_key="sk-test"
        ),
    )
    payload = settings_module.get_current_settings()
    assert payload["llm"] == {
        "provider": "anthropic",
        "model": "anthropic/claude",
        "configured": True,
        "structured_output": "unknown",
        "instructor_mode": "unknown",
    }


def test_structured_output_path_leaves_as_a_closed_value(monkeypatch):
    """Which path obtains structured output is where a schema rejection shows up;
    the setting leaves as one of the shipped frameworks, ``invalid``, or ``unknown``."""
    from types import SimpleNamespace

    cases = [
        ("BAML", "json_schema_mode", "baml", "json_schema_mode"),
        ("litellm_native", "", "litellm_native", "default"),
        ("instructor", "Tool Call; drop table", "instructor", "invalid"),
        ("my_private_framework", None, "invalid", "unknown"),
    ]
    for framework, mode, expected_framework, expected_mode in cases:
        config = SimpleNamespace(
            llm_provider="openai",
            llm_model="openai/gpt-5-mini",
            llm_api_key="sk-test",
            structured_output_framework=framework,
        )
        if mode is not None:
            config.llm_instructor_mode = mode
        monkeypatch.setattr(settings_module, "get_llm_context_config", lambda config=config: config)

        llm = settings_module.get_current_settings()["llm"]

        assert llm["structured_output"] == expected_framework, framework
        assert llm["instructor_mode"] == expected_mode, mode
        assert "drop table" not in repr(llm)


def test_configured_follows_the_per_call_config_not_the_process_one(monkeypatch):
    """No mock of llm_available: a per-call config with a key makes the event
    report a usable LLM, and one without a key an unused default."""
    from types import SimpleNamespace

    for key, expected in (("sk-test", True), ("", False)):
        monkeypatch.setattr(
            settings_module,
            "get_llm_context_config",
            lambda key=key: SimpleNamespace(
                llm_provider="openai", llm_model="openai/gpt-5-mini", llm_api_key=key
            ),
        )
        assert settings_module.get_current_settings()["llm"]["configured"] is expected
