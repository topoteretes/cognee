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
    REASON_NO_WRITES_SINCE_LAST_IMPROVE,
    REASON_OPT_IN_DISABLED,
    REASON_PERSONALIZATION_DISABLED,
    REASON_TURNED_OFF,
    BuildTruthSubspaceStage,
    DistillSessionsStage,
    FeedbackWeightsStage,
    GlobalContextIndexStage,
    PersistSessionQAStage,
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


# --- stage 5 ---------------------------------------------------------------


def _distilled(status="completed"):
    return types.SimpleNamespace(status=status, documents=["lesson"], uncurated_entries=0)


@pytest.mark.asyncio
async def test_distill_stops_at_the_first_exhausted_budget(monkeypatch):
    """A 402 on one session ends the loop: every later session would fail the same
    way, and the stage result must still classify the run as budget-exhausted."""
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

    distill_mod = importlib.import_module("cognee.modules.session_distillation")
    distill = AsyncMock(side_effect=[_distilled(), LLMPaymentRequiredError(), _distilled()])
    monkeypatch.setattr(distill_mod, "distill_session", distill)

    result = await DistillSessionsStage().run(_inputs(session_ids=("a", "b", "c")))

    assert [call.args[0] for call in distill.await_args_list] == ["a", "b"]
    assert result.status == "errored"
    assert result.budget_exhausted
    assert result.counts["sessions_completed"] == 1
    assert result.counts["sessions_failed"] == 1


@pytest.mark.asyncio
async def test_distill_continues_past_an_ordinary_session_failure(monkeypatch):
    distill_mod = importlib.import_module("cognee.modules.session_distillation")
    distill = AsyncMock(side_effect=[RuntimeError("one session broke"), _distilled()])
    monkeypatch.setattr(distill_mod, "distill_session", distill)

    result = await DistillSessionsStage().run(_inputs(session_ids=("a", "b")))

    assert distill.await_count == 2
    assert result.status == "errored"
    assert not result.budget_exhausted
    assert result.counts["sessions_completed"] == 1


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


# --- stage 9 ---------------------------------------------------------------


def test_global_context_index_skipped_when_turned_off():
    stage = GlobalContextIndexStage()
    assert stage.gate(_inputs()) == REASON_TURNED_OFF
    assert stage.gate(_inputs(build_global_context_index=True)) is None


def _patch_global_context_deps(monkeypatch, *, changed, pipeline_returns, shared_graph=False):
    changes_mod = importlib.import_module("cognee.modules.improve.watermarks")
    check = AsyncMock(return_value=changed)
    monkeypatch.setattr(changes_mod, "has_dataset_changed_since_last_improve", check)
    access_mod = importlib.import_module("cognee.context_global_variables")
    monkeypatch.setattr(access_mod, "backend_access_control_enabled", lambda: not shared_graph)
    pipeline_mod = importlib.import_module("cognee.memify_pipelines.global_context_index")
    pipeline = AsyncMock(return_value=pipeline_returns)
    monkeypatch.setattr(pipeline_mod, "global_context_index_pipeline", pipeline)
    return check, pipeline


@pytest.mark.asyncio
async def test_global_context_index_skips_the_build_when_only_reads_ran(monkeypatch):
    from cognee.modules.improve.watermarks import GLOBAL_CONTEXT_WATERMARK_KEY

    check, pipeline = _patch_global_context_deps(monkeypatch, changed=False, pipeline_returns={})
    inputs = _inputs(build_global_context_index=True)

    result = await GlobalContextIndexStage().run(inputs)

    pipeline.assert_not_awaited()
    assert result.status == "already_completed"
    assert result.reason == REASON_NO_WRITES_SINCE_LAST_IMPROVE
    assert result.run_info_stamp[GLOBAL_CONTEXT_WATERMARK_KEY]["status"] == "already_completed"
    assert check.await_args.args == (inputs.dataset_id, GLOBAL_CONTEXT_WATERMARK_KEY)
    assert check.await_args.kwargs["any_dataset"] is False


@pytest.mark.asyncio
async def test_global_context_index_stamps_a_completed_build_but_not_an_errored_one(monkeypatch):
    from cognee.modules.improve.watermarks import GLOBAL_CONTEXT_WATERMARK_KEY
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored

    _, pipeline = _patch_global_context_deps(monkeypatch, changed=True, pipeline_returns={"ok": 1})
    built = await GlobalContextIndexStage().run(_inputs(build_global_context_index=True))
    pipeline.assert_awaited_once()
    assert built.run_info_stamp[GLOBAL_CONTEXT_WATERMARK_KEY]["status"] == "completed"

    errored_run = PipelineRunErrored(
        pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="docs", error_message="bad"
    )
    _patch_global_context_deps(monkeypatch, changed=True, pipeline_returns=errored_run)
    errored = await GlobalContextIndexStage().run(_inputs(build_global_context_index=True))
    assert errored.status == "errored"
    assert errored.run_info_stamp is None


@pytest.mark.asyncio
async def test_global_context_index_counts_every_dataset_when_the_graph_is_shared(monkeypatch):
    """Relation and type lines read edges that any dataset in a shared graph may write."""
    check, _ = _patch_global_context_deps(
        monkeypatch, changed=False, pipeline_returns={}, shared_graph=True
    )

    await GlobalContextIndexStage().run(_inputs(build_global_context_index=True))

    assert check.await_args.kwargs["any_dataset"] is True


# --- inputs ----------------------------------------------------------------


def test_inputs_are_frozen():
    inputs = _inputs()
    with pytest.raises(FrozenInstanceError):
        inputs.dataset_id = uuid4()  # type: ignore[misc]
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
