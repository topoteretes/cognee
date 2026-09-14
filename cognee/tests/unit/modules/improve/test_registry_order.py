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
    "triplet_enrichment",
    "global_context_index",
]


def _position(name: str) -> int:
    return stage_names().index(name)


def test_registry_lists_the_nine_stages_in_plan_order():
    """The registry list is the single declaration of the order (no ``after`` mirror)."""
    assert stage_names(DEFAULT_STAGES) == EXPECTED_ORDER


def test_extract_agent_context_runs_before_distill_sessions():
    assert _position("extract_agent_context") < _position("distill_sessions")


def test_distill_sessions_runs_before_build_truth_subspace():
    assert _position("distill_sessions") < _position("build_truth_subspace")


def test_build_truth_subspace_runs_before_triplet_enrichment():
    assert _position("build_truth_subspace") < _position("triplet_enrichment")


def test_persist_session_qa_is_the_only_fatal_stage():
    fatal = [stage.name for stage in DEFAULT_STAGES if stage.fatal]
    assert fatal == ["persist_session_qa"]
    validate_fatal_stage_policy(DEFAULT_STAGES)


def test_session_fed_stages_and_graph_stages():
    needs = {stage.name: stage.needs_sessions for stage in DEFAULT_STAGES}
    assert needs["triplet_enrichment"] is False
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
    validate_stages_disabled(["triplet_enrichment", "distill_sessions"])


def test_validate_stages_disabled_rejects_unknown_and_fatal_names():
    with pytest.raises(ValueError, match="persist_sesion_qa"):
        validate_stages_disabled(["persist_sesion_qa"])  # one typo, named in the error
    with pytest.raises(ValueError, match="fatal"):
        validate_stages_disabled(["persist_session_qa"])
