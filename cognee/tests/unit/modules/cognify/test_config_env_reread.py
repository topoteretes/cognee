"""get_cognify_config() re-reads environment variables on every call (issue #5771)."""

from cognee.modules.cognify import config as cognify_config
from cognee.modules.cognify.config import get_cognify_config, override_cognify_config


def test_env_change_between_calls_is_picked_up(monkeypatch):
    """The issue's repro: GRAPH_EXTRACTOR flipped between two calls."""
    monkeypatch.setenv("GRAPH_EXTRACTOR", "llm")
    assert get_cognify_config().graph_extractor == "llm"

    monkeypatch.setenv("GRAPH_EXTRACTOR", "gliner_demo")
    assert get_cognify_config().graph_extractor == "gliner_demo"


def test_provenance_env_change_between_calls_is_picked_up(monkeypatch):
    monkeypatch.setenv("PROVENANCE_TRACKING", "true")
    assert get_cognify_config().provenance_tracking is True

    monkeypatch.setenv("PROVENANCE_TRACKING", "false")
    assert get_cognify_config().provenance_tracking is False


def test_calls_do_not_share_mutable_state(monkeypatch):
    """Mutating one returned config must not leak into later calls."""
    monkeypatch.delenv("GRAPH_EXTRACTOR", raising=False)
    first = get_cognify_config()
    first.graph_extractor = "llm"  # the default is "auto", so this is a change

    second = get_cognify_config()
    assert second.graph_extractor == "auto"


def test_override_takes_precedence_over_env(monkeypatch):
    """cognee.config setters keep their process-wide effect via overrides."""
    monkeypatch.setattr(cognify_config, "_config_overrides", {})
    monkeypatch.setenv("PROVENANCE_TRACKING", "false")

    override_cognify_config(provenance_tracking=True)
    assert get_cognify_config().provenance_tracking is True
