"""ImproveConfig owns only the loop's own knobs (plan Part 5.9)."""

import pytest
from pydantic import ValidationError

from cognee.modules.improve import DEFAULT_FEEDBACK_ALPHA, ImproveConfig, get_improve_config


@pytest.fixture(autouse=True)
def _clear_cache():
    get_improve_config.cache_clear()
    yield
    get_improve_config.cache_clear()


def test_defaults(monkeypatch):
    for name in (
        "IMPROVE_AUTO_ENABLED",
        "IMPROVE_DEBOUNCE_ENTRIES",
        "IMPROVE_DEBOUNCE_SECONDS",
        "IMPROVE_STAGES_DISABLED",
        "IMPROVE_FEEDBACK_ALPHA",
    ):
        monkeypatch.delenv(name, raising=False)

    config = ImproveConfig(_env_file=None)

    assert config.auto_enabled is True
    assert config.debounce_entries == 1
    assert config.debounce_seconds == 0.0
    assert config.stages_disabled == []
    assert config.feedback_alpha == DEFAULT_FEEDBACK_ALPHA == 0.1


def test_env_prefix_and_csv_parsing(monkeypatch):
    monkeypatch.setenv("IMPROVE_AUTO_ENABLED", "false")
    monkeypatch.setenv("IMPROVE_DEBOUNCE_ENTRIES", "5")
    monkeypatch.setenv("IMPROVE_DEBOUNCE_SECONDS", "2.5")
    monkeypatch.setenv("IMPROVE_STAGES_DISABLED", " feedback_weights, distill_sessions ,,")
    monkeypatch.setenv("IMPROVE_FEEDBACK_ALPHA", "0.25")

    config = ImproveConfig(_env_file=None)

    assert config.auto_enabled is False
    assert config.debounce_entries == 5
    assert config.debounce_seconds == 2.5
    assert config.stages_disabled == ["feedback_weights", "distill_sessions"]
    assert config.feedback_alpha == 0.25


def test_stages_disabled_accepts_a_list_directly():
    assert ImproveConfig(_env_file=None, stages_disabled=["a", " b "]).stages_disabled == [
        "a",
        "b",
    ]


@pytest.mark.parametrize("alpha", [0, -0.1, 1.0001, 5])
def test_feedback_alpha_must_be_in_unit_interval(alpha):
    with pytest.raises(ValidationError):
        ImproveConfig(_env_file=None, feedback_alpha=alpha)


def test_feedback_alpha_upper_bound_is_inclusive():
    assert ImproveConfig(_env_file=None, feedback_alpha=1).feedback_alpha == 1


def test_negative_debounce_rejected():
    with pytest.raises(ValidationError):
        ImproveConfig(_env_file=None, debounce_entries=-1)
    with pytest.raises(ValidationError):
        ImproveConfig(_env_file=None, debounce_seconds=-0.5)


def test_config_declares_no_shared_knobs():
    fields = set(ImproveConfig.model_fields)
    assert fields == {
        "auto_enabled",
        "debounce_entries",
        "debounce_seconds",
        "stages_disabled",
        "feedback_alpha",
    }
    for shared in ("triplet_embedding", "caching", "auto_feedback", "personalization_enabled"):
        assert shared not in fields


def test_getter_is_cached():
    assert get_improve_config() is get_improve_config()
