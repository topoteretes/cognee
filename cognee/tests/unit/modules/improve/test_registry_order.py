"""Pins the improve registry: names, order, ``after`` dependencies, the one fatal stage."""

import pytest

from cognee.modules.improve import DEFAULT_STAGES, stage_names, validate_stage_order
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
    assert stage_names(DEFAULT_STAGES) == EXPECTED_ORDER


def test_registry_order_satisfies_after_declarations():
    validate_stage_order(DEFAULT_STAGES)


def test_extract_agent_context_runs_before_distill_sessions():
    assert _position("extract_agent_context") < _position("distill_sessions")
    assert "extract_agent_context" in _stage("distill_sessions").after


def test_distill_sessions_runs_before_build_truth_subspace():
    assert _position("distill_sessions") < _position("build_truth_subspace")
    assert "distill_sessions" in _stage("build_truth_subspace").after


def test_build_truth_subspace_runs_before_triplet_enrichment():
    assert _position("build_truth_subspace") < _position("triplet_enrichment")
    assert "build_truth_subspace" in _stage("triplet_enrichment").after


def test_persist_session_qa_is_the_only_fatal_stage():
    fatal = [stage.name for stage in DEFAULT_STAGES if stage.fatal]
    assert fatal == ["persist_session_qa"]


def test_session_stages_and_graph_stages():
    kinds = {stage.name: stage.kind for stage in DEFAULT_STAGES}
    assert kinds["triplet_enrichment"] == "graph"
    assert kinds["global_context_index"] == "graph"
    assert all(kinds[name] == "session" for name in EXPECTED_ORDER[:7])


def _stage(name):
    return next(stage for stage in DEFAULT_STAGES if stage.name == name)


class _Stage(BaseStage):
    def __init__(self, name, after=(), fatal=False):
        self.name = name
        self.after = tuple(after)
        self.fatal = fatal


def test_validate_rejects_a_stage_listed_before_its_dependency():
    stages = [_Stage("b", after=("a",)), _Stage("a"), _Stage("persist_session_qa", fatal=True)]
    with pytest.raises(ValueError, match="must run after"):
        validate_stage_order(stages)


def test_validate_rejects_unknown_dependency_and_duplicates():
    with pytest.raises(ValueError, match="not registered"):
        validate_stage_order(
            [_Stage("a", after=("ghost",)), _Stage("persist_session_qa", fatal=True)]
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_stage_order([_Stage("a"), _Stage("a"), _Stage("persist_session_qa", fatal=True)])


def test_validate_requires_exactly_one_fatal_stage():
    with pytest.raises(ValueError, match="fatal"):
        validate_stage_order([_Stage("a"), _Stage("b")])
    with pytest.raises(ValueError, match="fatal"):
        validate_stage_order([_Stage("a", fatal=True), _Stage("persist_session_qa", fatal=True)])
