"""Pins the improve registry: names, order, the one fatal stage, session-fed stages."""

import pytest

from cognee.modules.improve import (
    DEFAULT_STAGES,
    stage_names,
    validate_fatal_stage_policy,
    validate_stages,
    validate_stages_disabled,
)
from cognee.modules.improve.stage import BaseStage

EXPECTED_ORDER = [
    "feedback_weights",
    "persist_session_qa",
    "persist_agent_traces",
    "extract_agent_context",
    "distill_sessions",
    "update_user_preferences",
    "build_truth_subspace",
    "review_conflicts",
    "global_context_index",
]


def _position(name: str) -> int:
    return stage_names().index(name)


def test_registry_lists_stages_in_plan_order():
    """The registry list is the single declaration of the order (no ``after`` mirror)."""
    assert stage_names(DEFAULT_STAGES) == EXPECTED_ORDER


def test_extract_agent_context_runs_before_distill_sessions():
    assert _position("extract_agent_context") < _position("distill_sessions")


def test_distill_sessions_runs_before_build_truth_subspace():
    assert _position("distill_sessions") < _position("build_truth_subspace")


def test_build_truth_subspace_runs_before_review_conflicts():
    assert _position("build_truth_subspace") < _position("review_conflicts")


def test_persist_session_qa_is_the_only_fatal_stage():
    fatal = [stage.name for stage in DEFAULT_STAGES if stage.fatal]
    assert fatal == ["persist_session_qa"]
    validate_fatal_stage_policy(DEFAULT_STAGES)


def test_session_fed_stages_and_graph_stages():
    needs = {stage.name: stage.needs_sessions for stage in DEFAULT_STAGES}
    assert needs["review_conflicts"] is False
    assert needs["global_context_index"] is False
    assert all(needs[name] for name in EXPECTED_ORDER[:7])


class _Stage(BaseStage):
    def __init__(self, name, fatal=False):
        self.name = name
        self.fatal = fatal


def test_validate_rejects_nameless_and_duplicate_stages():
    with pytest.raises(ValueError, match="no name"):
        validate_stages([_Stage("a"), _Stage("")])
    with pytest.raises(ValueError, match="duplicate"):
        validate_stages([_Stage("a"), _Stage("a")])


def test_fatal_policy_requires_exactly_persist_session_qa():
    with pytest.raises(ValueError, match="fatal"):
        validate_fatal_stage_policy([_Stage("a"), _Stage("b")])
    with pytest.raises(ValueError, match="fatal"):
        validate_fatal_stage_policy(
            [_Stage("a", fatal=True), _Stage("persist_session_qa", fatal=True)]
        )
    validate_fatal_stage_policy([_Stage("a"), _Stage("persist_session_qa", fatal=True)])


def test_validate_stages_disabled_accepts_real_non_fatal_names():
    validate_stages_disabled([])
    validate_stages_disabled(["review_conflicts", "distill_sessions"])


def test_validate_stages_disabled_ignores_the_removed_triplet_enrichment():
    """A config written before SDK-1115 must not fail the boot, or any later improve()."""
    validate_stages_disabled(iter(["triplet_enrichment", "distill_sessions"]))
    assert "triplet_enrichment" not in stage_names()


def test_removed_stage_name_warns_once_per_process(monkeypatch):
    """improve() re-validates on every run; only the cached config read warns."""
    from unittest.mock import MagicMock

    from cognee.modules.improve import config as improve_config

    monkeypatch.setenv("IMPROVE_STAGES_DISABLED", "triplet_enrichment,distill_sessions")
    warning = MagicMock()
    monkeypatch.setattr(improve_config.logger, "warning", warning)
    improve_config.get_improve_config.cache_clear()
    try:
        config = improve_config.get_improve_config()
        improve_config.get_improve_config()
        validate_stages_disabled(config.stages_disabled)
        validate_stages_disabled(config.stages_disabled)
    finally:
        improve_config.get_improve_config.cache_clear()
    warning.assert_called_once()
    assert "triplet_enrichment" in str(warning.call_args)


def test_validate_stages_disabled_rejects_unknown_and_fatal_names():
    with pytest.raises(ValueError, match="persist_sesion_qa"):
        validate_stages_disabled(["persist_sesion_qa"])  # one typo, named in the error
    with pytest.raises(ValueError, match="fatal"):
        validate_stages_disabled(["persist_session_qa"])
