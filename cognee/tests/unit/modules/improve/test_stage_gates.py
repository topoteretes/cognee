"""The free gates of the real stages (plan A2), with every dependency stubbed."""

import importlib
import types
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.improve import (
    REASON_BACKEND_UNSUPPORTED,
    REASON_DISABLED_BY_CONFIG,
    REASON_NO_SESSION_IDS,
    GraphCapabilities,
    ImproveConfig,
    ImproveRunInputs,
    evaluate_gate,
)
from cognee.modules.improve.stages import (
    REASON_FEEDBACK_INFLUENCE_ZERO,
    REASON_OPT_IN_DISABLED,
    REASON_PERSONALIZATION_DISABLED,
    REASON_TRIPLET_EMBEDDING_DISABLED,
    BuildTruthSubspaceStage,
    FeedbackWeightsStage,
    GlobalContextIndexStage,
    PersistSessionQAStage,
    TripletEnrichmentStage,
    UpdateUserPreferencesStage,
)


def _inputs(session_ids=("s1",), capabilities=None, config=None, **overrides):
    user = types.SimpleNamespace(id=uuid4())
    dataset = types.SimpleNamespace(id=uuid4(), name="docs", owner_id=user.id)
    return ImproveRunInputs(
        user=user,
        dataset_id=dataset.id,
        dataset=dataset,
        session_ids=tuple(session_ids),
        config=config or ImproveConfig(),
        capabilities=capabilities or GraphCapabilities.assume_supported(),
        **overrides,
    )


def _patch_base_config(monkeypatch, **fields):
    base_mod = importlib.import_module("cognee.base_config")
    defaults = {"default_feedback_influence": 0.0, "personalization_enabled": False}
    defaults.update(fields)
    monkeypatch.setattr(base_mod, "get_base_config", lambda: types.SimpleNamespace(**defaults))


def _patch_cognify_config(monkeypatch, triplet_embedding):
    mod = importlib.import_module("cognee.modules.cognify.config")
    monkeypatch.setattr(
        mod,
        "get_cognify_config",
        lambda: types.SimpleNamespace(triplet_embedding=triplet_embedding),
    )


# --- stage 1 ---------------------------------------------------------------


def test_feedback_weights_skipped_at_zero_influence(monkeypatch):
    _patch_base_config(monkeypatch, default_feedback_influence=0.0)
    assert FeedbackWeightsStage().gate(_inputs()) == REASON_FEEDBACK_INFLUENCE_ZERO


def test_feedback_weights_skipped_on_unsupported_backend(monkeypatch):
    _patch_base_config(monkeypatch, default_feedback_influence=0.5)
    caps = GraphCapabilities(supports_feedback_weights=False, supports_truth_state=False)
    assert FeedbackWeightsStage().gate(_inputs(capabilities=caps)) == REASON_BACKEND_UNSUPPORTED


def test_feedback_weights_runs_when_influence_and_backend_allow(monkeypatch):
    _patch_base_config(monkeypatch, default_feedback_influence=0.5)
    assert FeedbackWeightsStage().gate(_inputs()) is None


def test_session_stage_without_sessions_is_skipped_before_its_own_gate(monkeypatch):
    _patch_base_config(monkeypatch, default_feedback_influence=0.5)
    assert evaluate_gate(FeedbackWeightsStage(), _inputs(session_ids=())) == REASON_NO_SESSION_IDS


def test_disabled_by_config_wins_over_everything():
    config = ImproveConfig(stages_disabled=["persist_session_qa"])
    assert (
        evaluate_gate(PersistSessionQAStage(), _inputs(config=config)) == REASON_DISABLED_BY_CONFIG
    )


@pytest.mark.asyncio
async def test_feedback_weights_passes_resolved_id_and_alpha(monkeypatch):
    pipeline_mod = importlib.import_module("cognee.memify_pipelines.apply_feedback_weights")
    pipeline = AsyncMock(return_value={})
    monkeypatch.setattr(pipeline_mod, "apply_feedback_weights_pipeline", pipeline)
    inputs = _inputs(session_ids=("a", "b"), feedback_alpha=0.4)

    result = await FeedbackWeightsStage().run(inputs)

    pipeline.assert_awaited_once_with(
        user=inputs.user,
        session_ids=["a", "b"],
        dataset=inputs.dataset_id,
        alpha=0.4,
        run_in_background=False,
    )
    assert result.status == "completed"
    assert result.counts == {"sessions": 2}


# --- stage 6 ---------------------------------------------------------------


def test_user_preferences_skipped_when_personalization_off(monkeypatch):
    _patch_base_config(monkeypatch, personalization_enabled=False)
    assert UpdateUserPreferencesStage().gate(_inputs()) == REASON_PERSONALIZATION_DISABLED


@pytest.mark.asyncio
async def test_user_preferences_maps_disabled_status_to_skipped(monkeypatch):
    _patch_base_config(monkeypatch, personalization_enabled=True)
    update_mod = importlib.import_module("cognee.modules.user_preferences.update")
    monkeypatch.setattr(
        update_mod,
        "update_user_preferences",
        AsyncMock(
            return_value=update_mod.PreferenceUpdateResult(status="personalization_disabled")
        ),
    )
    result = await UpdateUserPreferencesStage().run(_inputs())
    assert result.status == "skipped"
    assert result.reason == REASON_PERSONALIZATION_DISABLED


@pytest.mark.asyncio
async def test_user_preferences_completed_carries_counts(monkeypatch):
    update_mod = importlib.import_module("cognee.modules.user_preferences.update")
    fake = AsyncMock(
        return_value=update_mod.PreferenceUpdateResult(
            status="completed", turns_applied=2, edges_written=3, edges_pruned=1, text_lines_added=4
        )
    )
    monkeypatch.setattr(update_mod, "update_user_preferences", fake)
    inputs = _inputs(session_ids=("s1", "s2"))

    result = await UpdateUserPreferencesStage().run(inputs)

    fake.assert_awaited_once_with(
        session_ids=["s1", "s2"], dataset=inputs.dataset_id, user=inputs.user
    )
    assert result.status == "completed"
    assert result.counts == {
        "turns_applied": 2,
        "edges_written": 3,
        "edges_pruned": 1,
        "text_lines_added": 4,
    }


# --- stage 7 ---------------------------------------------------------------


def test_truth_subspace_skipped_unless_opted_in():
    assert BuildTruthSubspaceStage().gate(_inputs()) == REASON_OPT_IN_DISABLED


def test_truth_subspace_skipped_on_backend_without_truth_state():
    caps = GraphCapabilities(supports_feedback_weights=True, supports_truth_state=False)
    stage = BuildTruthSubspaceStage()
    assert stage.gate(_inputs(build_truth_subspace=True, capabilities=caps)) == (
        REASON_BACKEND_UNSUPPORTED
    )
    assert stage.gate(_inputs(build_truth_subspace=True)) is None


@pytest.mark.asyncio
async def test_truth_subspace_passes_resolved_id(monkeypatch):
    build_mod = importlib.import_module("cognee.modules.truth_subspace.build")
    fake = AsyncMock(
        return_value={"anchors": 2, "nodes_scored": 10, "signature": "x", "truth_epoch": 3}
    )
    monkeypatch.setattr(build_mod, "build_truth_subspace", fake)
    inputs = _inputs(build_truth_subspace=True)

    result = await BuildTruthSubspaceStage().run(inputs)

    fake.assert_awaited_once_with(dataset=inputs.dataset_id, session_ids=["s1"], user=inputs.user)
    assert result.status == "completed"
    assert result.counts == {"anchors": 2, "nodes_scored": 10, "truth_epoch": 3}


# --- stage 8 ---------------------------------------------------------------


def test_triplet_enrichment_skipped_when_triplet_embedding_off(monkeypatch):
    _patch_cognify_config(monkeypatch, triplet_embedding=False)
    assert TripletEnrichmentStage().gate(_inputs()) == REASON_TRIPLET_EMBEDDING_DISABLED


def test_triplet_enrichment_runs_with_flag_on(monkeypatch):
    _patch_cognify_config(monkeypatch, triplet_embedding=True)
    assert TripletEnrichmentStage().gate(_inputs()) is None


def test_triplet_enrichment_custom_tasks_bypass_the_flag(monkeypatch):
    _patch_cognify_config(monkeypatch, triplet_embedding=False)
    inputs = _inputs(memify_kwargs={"enrichment_tasks": [object()]})
    assert TripletEnrichmentStage().gate(inputs) is None


@pytest.mark.asyncio
async def test_triplet_enrichment_custom_tasks_skip_the_change_check(monkeypatch):
    changes_mod = importlib.import_module("cognee.modules.improve.graph_changes")
    changed = AsyncMock(return_value=False)
    monkeypatch.setattr(changes_mod, "has_graph_changed_since_last_improve", changed)
    memify_mod = importlib.import_module("cognee.modules.memify")
    memify = AsyncMock(return_value={"ok": 1})
    monkeypatch.setattr(memify_mod, "memify", memify)
    tasks = [object()]
    inputs = _inputs(memify_kwargs={"extraction_tasks": tasks}, node_name=["n"])

    result = await TripletEnrichmentStage().run(inputs)

    changed.assert_not_awaited()
    memify.assert_awaited_once()
    kwargs = memify.await_args.kwargs
    assert kwargs["extraction_tasks"] == tasks
    assert kwargs["dataset"] == inputs.dataset_id
    assert kwargs["node_name"] == ["n"]
    assert kwargs["run_in_background"] is False
    assert "data" not in kwargs
    assert result.raw_run == {"ok": 1}


# --- stage 9 ---------------------------------------------------------------


def test_global_context_index_skipped_unless_opted_in():
    stage = GlobalContextIndexStage()
    assert stage.gate(_inputs()) == REASON_OPT_IN_DISABLED
    assert stage.gate(_inputs(build_global_context_index=True)) is None


# --- inputs ----------------------------------------------------------------


def test_inputs_are_frozen():
    inputs = _inputs()
    with pytest.raises(Exception):
        inputs.dataset_id = uuid4()  # type: ignore[misc]
    with pytest.raises(TypeError):
        inputs.memify_kwargs["data"] = 1  # type: ignore[index]
    assert not hasattr(inputs, "run_in_background")
