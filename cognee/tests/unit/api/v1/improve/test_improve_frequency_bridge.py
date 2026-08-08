import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def improve_mod():
    return importlib.import_module("cognee.api.v1.improve.improve")


@pytest.mark.asyncio
async def test_bridge_sessions_invokes_frequency_pipeline(monkeypatch, improve_mod):
    calls = []
    user = SimpleNamespace(id="u1")

    async def fake_resolve_dataset_name(dataset, resolved_user):
        assert dataset == "main_dataset"
        assert resolved_user is user
        return "resolved_dataset"

    async def fake_feedback_pipeline(**kwargs):
        calls.append(("feedback", kwargs))

    async def fake_frequency_pipeline(**kwargs):
        calls.append(("frequency", kwargs))

    async def fake_persist_pipeline(**kwargs):
        calls.append(("persist", kwargs))

    feedback_mod = importlib.import_module("cognee.memify_pipelines.apply_feedback_weights")
    frequency_mod = importlib.import_module("cognee.memify_pipelines.apply_frequency_weights")
    persist_mod = importlib.import_module(
        "cognee.memify_pipelines.persist_sessions_in_knowledge_graph"
    )

    monkeypatch.setattr(improve_mod, "_resolve_dataset_name", fake_resolve_dataset_name)
    monkeypatch.setattr(feedback_mod, "apply_feedback_weights_pipeline", fake_feedback_pipeline)
    monkeypatch.setattr(frequency_mod, "apply_frequency_weights_pipeline", fake_frequency_pipeline)
    monkeypatch.setattr(
        persist_mod,
        "persist_sessions_in_knowledge_graph_pipeline",
        fake_persist_pipeline,
    )

    await improve_mod._bridge_sessions(
        dataset="main_dataset",
        session_ids=["s1"],
        user=user,
        feedback_alpha=0.2,
        run_in_background=False,
    )

    assert [call[0] for call in calls] == ["feedback", "frequency", "persist"]
    assert calls[0][1]["alpha"] == 0.2
    assert calls[1][1] == {
        "user": user,
        "session_ids": ["s1"],
        "dataset": "resolved_dataset",
        "run_in_background": False,
    }
