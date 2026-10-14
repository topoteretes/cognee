"""``remember(codegraph_config=...)`` reaches cognify(), where the CODE / CODE_REPO
routes read ``index_vectors``; it is not an add() option and it never applies on
the session path, which runs no cognify.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

remember_module = importlib.import_module("cognee.api.v1.remember.remember")


@pytest.fixture(autouse=True)
def _no_db_setup(monkeypatch):
    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)


@pytest.fixture
def permanent_pipeline(monkeypatch):
    """Stub add()/cognify() so the permanent path runs without databases."""
    calls = {}

    async def fake_add(*args, **kwargs):
        calls["add"] = kwargs

    async def fake_cognify(*args, **kwargs):
        calls["cognify"] = kwargs
        return {}

    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)
    return calls


@pytest.mark.asyncio
async def test_codegraph_config_reaches_cognify_not_add(permanent_pipeline):
    config = {"index_vectors": True}

    result = await remember_module.remember(
        "/some/repo",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        codegraph_config=config,
    )

    assert result.status == "completed"
    assert permanent_pipeline["cognify"]["codegraph_config"] == config
    assert "codegraph_config" not in permanent_pipeline["add"]


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_session_id(permanent_pipeline):
    # The session path never runs cognify, so the option would be ignored.
    with pytest.raises(ValueError, match="codegraph_config is not supported when session_id"):
        await remember_module.remember(
            "note",
            session_id="s1",
            user=SimpleNamespace(id=uuid4()),
            codegraph_config={"index_vectors": True},
        )

    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_code_content_type(permanent_pipeline):
    # content_type="code" runs its own pipeline and never reaches cognify().
    with pytest.raises(ValueError, match="codegraph_config is not supported with content_type"):
        await remember_module.remember(
            "/some/repo",
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            content_type="code",
            codegraph_config={"index_vectors": True},
        )

    assert permanent_pipeline == {}
