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
    assert config.distill_max_batches_per_run == 0


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


def test_negative_distill_cap_rejected():
    with pytest.raises(ValueError, match="distill_max_batches_per_run"):
        ImproveConfig(_env_file=None, distill_max_batches_per_run=-1)


def test_distill_cap_reads_env(monkeypatch):
    monkeypatch.setenv("IMPROVE_DISTILL_MAX_BATCHES_PER_RUN", "3")
    assert ImproveConfig(_env_file=None).distill_max_batches_per_run == 3


def test_config_declares_no_shared_knobs():
    fields = set(ImproveConfig.model_fields)
    assert fields == {
        "auto_enabled",
        "debounce_entries",
        "debounce_seconds",
        "stages_disabled",
        "feedback_alpha",
        "review_conflicts",
        "global_context_index",
        "effective_date_key",
        "trace_window_steps",
        "trace_windows_per_run",
        "distill_max_batches_per_run",
        "distillation_mode",
    }
    for shared in ("triplet_embedding", "caching", "auto_feedback", "personalization_enabled"):
        assert shared not in fields


def test_getter_is_cached():
    assert get_improve_config() is get_improve_config()


def test_distillation_mode_reads_the_bare_env_name(monkeypatch):
    monkeypatch.delenv("DISTILLATION_MODE", raising=False)
    assert ImproveConfig(_env_file=None).distillation_mode == ""
    assert ImproveConfig(_env_file=None).coding_distillation is False

    monkeypatch.setenv("DISTILLATION_MODE", " Coding ")
    config = ImproveConfig(_env_file=None)
    assert config.distillation_mode == "coding"
    assert config.coding_distillation is True

    monkeypatch.setenv("IMPROVE_DISTILLATION_MODE", "coding")
    monkeypatch.delenv("DISTILLATION_MODE")
    assert ImproveConfig(_env_file=None).distillation_mode == ""  # the prefix is not read


def test_distillation_mode_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("DISTILLATION_MODE", "research")
    with pytest.raises(ValidationError):
        ImproveConfig(_env_file=None)


def test_the_mode_has_one_source_for_the_whole_process(monkeypatch):
    """The stage gate reads ``inputs.config`` (the cached config) and the distillation
    and the per-trace hook read ``coding_distillation_enabled()``; they must agree, so
    the function reads the same cached config instead of the live env."""
    from cognee.modules.improve.config import coding_distillation_enabled

    monkeypatch.setenv("DISTILLATION_MODE", "coding")
    assert get_improve_config().coding_distillation and coding_distillation_enabled()

    monkeypatch.delenv("DISTILLATION_MODE")
    assert coding_distillation_enabled() is get_improve_config().coding_distillation is True

    get_improve_config.cache_clear()
    assert coding_distillation_enabled() is get_improve_config().coding_distillation is False
