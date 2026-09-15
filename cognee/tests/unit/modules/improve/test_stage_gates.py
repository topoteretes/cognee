"""The free gates of the real stages (plan A2), with every dependency stubbed."""

import importlib
import types
from dataclasses import FrozenInstanceError
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


def test_feedback_weights_runs_at_zero_global_influence(monkeypatch):
    """feedback_influence is a per-call read-time knob; the write must not gate on
    the global default (0 is the only value base_config doesn't warn against)."""
    _patch_base_config(monkeypatch, default_feedback_influence=0.0)
    assert FeedbackWeightsStage().gate(_inputs()) is None


def test_feedback_weights_skipped_on_unsupported_backend(monkeypatch):
    _patch_base_config(monkeypatch, default_feedback_influence=0.5)
    caps = GraphCapabilities(supports_feedback_weights=False, supports_truth_state=False)
    assert FeedbackWeightsStage().gate(_inputs(capabilities=caps)) == REASON_BACKEND_UNSUPPORTED


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


@pytest.mark.asyncio
async def test_truth_subspace_maps_an_errored_build_onto_the_stage(monkeypatch):
    """The build fails open — a failed write is a status dict, not a raise — so the
    stage must translate it: a completed report would hide the failure from the
    ImproveResult and keep the improve operation row at "succeeded"."""
    build_mod = importlib.import_module("cognee.modules.truth_subspace.build")
    fake = AsyncMock(
        return_value={
            "anchors": 2,
            "nodes_scored": 0,
            "nodes_skipped": 5,
            "signature": "x",
            "truth_epoch": 3,
            "status": "errored",
            "error": "centroid commit failed: boom",
        }
    )
    monkeypatch.setattr(build_mod, "build_truth_subspace", fake)

    result = await BuildTruthSubspaceStage().run(_inputs(build_truth_subspace=True))

    assert result.status == "errored"
    assert "centroid commit failed" in result.error
    assert result.counts["nodes_skipped"] == 5


@pytest.mark.asyncio
async def test_truth_subspace_maps_a_skipped_build_onto_the_stage(monkeypatch):
    """Normally unreachable past the stage gate, but the mapping must not lie."""
    build_mod = importlib.import_module("cognee.modules.truth_subspace.build")
    fake = AsyncMock(
        return_value={
            "anchors": 0,
            "nodes_scored": 0,
            "nodes_skipped": 0,
            "signature": "",
            "truth_epoch": 0,
            "status": "skipped",
            "reason": "backend_unsupported",
        }
    )
    monkeypatch.setattr(build_mod, "build_truth_subspace", fake)

    result = await BuildTruthSubspaceStage().run(_inputs(build_truth_subspace=True))

    assert result.status == "skipped"
    assert result.reason == "backend_unsupported"


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


def _enrichment_stamp(result):
    from cognee.modules.improve.graph_changes import ENRICHMENT_WATERMARK_KEY

    return (result.run_info_stamp or {}).get(ENRICHMENT_WATERMARK_KEY)


def _patch_enrichment_deps(monkeypatch, *, changed, memify_returns):
    changes_mod = importlib.import_module("cognee.modules.improve.graph_changes")
    monkeypatch.setattr(
        changes_mod, "has_graph_changed_since_last_improve", AsyncMock(return_value=changed)
    )
    memify_mod = importlib.import_module("cognee.modules.memify")
    monkeypatch.setattr(memify_mod, "memify", AsyncMock(return_value=memify_returns))


@pytest.mark.asyncio
async def test_triplet_enrichment_stamps_full_scope_runs(monkeypatch):
    """The watermark's write side: a full-scope stage 8 that enriched (or
    verified nothing changed) stamps its own START time, so a write racing
    the operation-row close stays visible to the next run's gate."""
    from datetime import datetime, timezone

    _patch_enrichment_deps(monkeypatch, changed=True, memify_returns={"ok": 1})
    before = datetime.now(timezone.utc)

    enriched = await TripletEnrichmentStage().run(_inputs())

    stamp = _enrichment_stamp(enriched)
    assert stamp["status"] == "completed"
    assert before <= datetime.fromisoformat(stamp["started_at"]) <= datetime.now(timezone.utc)

    _patch_enrichment_deps(monkeypatch, changed=False, memify_returns={"ok": 1})
    verified = await TripletEnrichmentStage().run(_inputs())
    assert _enrichment_stamp(verified)["status"] == "already_completed"


@pytest.mark.asyncio
async def test_triplet_enrichment_never_stamps_scoped_or_errored_runs(monkeypatch):
    """node_name / custom-task runs do narrower or different work, and an
    errored run enriched nothing; 'nothing changed since' for the whole
    dataset must not be inferred from any of them."""
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored

    _patch_enrichment_deps(monkeypatch, changed=True, memify_returns={"ok": 1})
    scoped = await TripletEnrichmentStage().run(_inputs(node_name=["only_this"]))
    assert scoped.run_info_stamp is None
    custom = await TripletEnrichmentStage().run(
        _inputs(memify_kwargs={"extraction_tasks": [object()]})
    )
    assert custom.run_info_stamp is None

    errored_run = PipelineRunErrored(
        pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="docs", error_message="bad"
    )
    _patch_enrichment_deps(monkeypatch, changed=True, memify_returns=errored_run)
    errored = await TripletEnrichmentStage().run(_inputs())
    assert errored.status == "errored"
    assert errored.run_info_stamp is None


# --- stage 9 ---------------------------------------------------------------


def test_global_context_index_skipped_unless_opted_in():
    stage = GlobalContextIndexStage()
    assert stage.gate(_inputs()) == REASON_OPT_IN_DISABLED
    assert stage.gate(_inputs(build_global_context_index=True)) is None


# --- inputs ----------------------------------------------------------------


def test_inputs_are_frozen():
    inputs = _inputs()
    with pytest.raises(FrozenInstanceError):
        inputs.dataset_id = uuid4()  # type: ignore[misc]
    with pytest.raises(TypeError):
        inputs.memify_kwargs["data"] = 1  # type: ignore[index]
    assert not hasattr(inputs, "run_in_background")


# --- persist stages' nothing-new pre-check ----------------------------------


class _CoveredSessionManager:
    """Every session fully covered: entries exist, watermarks match the counts."""

    is_available = True

    def __init__(self):
        self.entry = types.SimpleNamespace(question="q", answer="a")

    async def get_session(self, *, user_id, session_id=None, formatted=False):
        return [self.entry]

    async def get_agent_trace_count(self, *, user_id, session_id=None):
        return 1

    async def get_session_context_entries(self, *, user_id, session_id=None):
        # One watermark row per kind; both stages read through StateRowWatermark.
        return [
            {
                "id": "session_persist_watermark",
                "kind": "session_persist_watermark_state",
                "persisted_qa_count": 1,
            },
            {
                "id": "agent_trace_persist_watermark",
                "kind": "agent_trace_persist_watermark_state",
                "persisted_trace_count": 1,
            },
        ]


def _install_covered_manager(monkeypatch):
    manager = _CoveredSessionManager()
    sm_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(sm_module, "get_session_manager", lambda: manager)
    return manager


@pytest.mark.asyncio
async def test_persist_qa_reports_already_completed_without_running_the_pipeline(monkeypatch):
    """A covered session must not run memify: an unconditional run logs a
    completed memify_pipeline row even with nothing new, which the enrichment
    change-check counts as a graph write."""
    from cognee.modules.improve.stages import REASON_NO_NEW_SESSION_ENTRIES

    _install_covered_manager(monkeypatch)
    pipeline_mod = importlib.import_module(
        "cognee.memify_pipelines.persist_sessions_in_knowledge_graph"
    )
    pipeline = AsyncMock()
    monkeypatch.setattr(pipeline_mod, "persist_sessions_in_knowledge_graph_pipeline", pipeline)

    result = await PersistSessionQAStage().run(_inputs(session_ids=("s1",)))

    assert result.status == "already_completed"
    assert result.reason == REASON_NO_NEW_SESSION_ENTRIES
    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_traces_reports_already_completed_without_running_the_pipeline(monkeypatch):
    from cognee.modules.improve.stages import REASON_NO_NEW_TRACE_STEPS, PersistAgentTracesStage

    _install_covered_manager(monkeypatch)
    pipeline_mod = importlib.import_module(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph"
    )
    pipeline = AsyncMock()
    monkeypatch.setattr(
        pipeline_mod, "persist_agent_trace_feedbacks_in_knowledge_graph_pipeline", pipeline
    )

    result = await PersistAgentTracesStage().run(_inputs(session_ids=("s1",)))

    assert result.status == "already_completed"
    assert result.reason == REASON_NO_NEW_TRACE_STEPS
    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_precheck_fails_open_to_running_the_pipeline(monkeypatch):
    class BrokenManager:
        is_available = True

        async def get_session(self, **kwargs):
            raise RuntimeError("cache down")

    sm_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(sm_module, "get_session_manager", lambda: BrokenManager())
    pipeline_mod = importlib.import_module(
        "cognee.memify_pipelines.persist_sessions_in_knowledge_graph"
    )
    pipeline = AsyncMock(return_value={})
    monkeypatch.setattr(pipeline_mod, "persist_sessions_in_knowledge_graph_pipeline", pipeline)

    await PersistSessionQAStage().run(_inputs(session_ids=("s1",)))

    pipeline.assert_awaited_once()
