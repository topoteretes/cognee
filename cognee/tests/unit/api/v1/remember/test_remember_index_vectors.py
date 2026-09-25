"""``remember(index_vectors=...)`` is a cognify option, and code needs no content type.

Code repositories go through ``add()`` + ``cognify()`` like any other input
(SDK-793); ``index_vectors`` reaches ``cognify()``, which hands it to the
CODE / CODE_REPO task lists.
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
@pytest.mark.parametrize("index_vectors", [False, True])
async def test_index_vectors_reaches_cognify_not_add(permanent_pipeline, index_vectors):
    result = await remember_module.remember(
        "/some/repo",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        index_vectors=index_vectors,
    )

    assert result.status == "completed"
    assert permanent_pipeline["cognify"]["index_vectors"] is index_vectors
    assert "index_vectors" not in permanent_pipeline["add"]


@pytest.mark.asyncio
async def test_code_content_type_is_rejected(permanent_pipeline):
    with pytest.raises(ValueError, match="need no content_type"):
        await remember_module.remember(
            "/some/repo",
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            content_type="code",
        )

    assert permanent_pipeline == {}
