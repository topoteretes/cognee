"""``remember(codegraph_config=...)`` reaches both add() and cognify(), and code needs no
content type.

Code repositories go through ``add()`` + ``cognify()`` like any other input
(SDK-793). ``codegraph_config`` is handed to both: add() reads
``repo_credentials`` to clone private repository URLs, cognify() reads
``index_vectors`` for the CODE / CODE_REPO task lists. The top-level
``index_vectors=`` / ``repo_credentials=`` spellings fold into it.
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
async def test_codegraph_config_reaches_add_and_cognify(permanent_pipeline):
    config = {"index_vectors": True, "repo_credentials": "tok123"}

    result = await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        codegraph_config=config,
    )

    assert result.status == "completed"
    assert permanent_pipeline["add"]["codegraph_config"] == config
    assert permanent_pipeline["cognify"]["codegraph_config"] == config


@pytest.mark.asyncio
@pytest.mark.parametrize("index_vectors", [False, True])
async def test_top_level_spellings_fold_into_codegraph_config(permanent_pipeline, index_vectors):
    result = await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        index_vectors=index_vectors,
        repo_credentials="tok123",
    )

    assert result.status == "completed"
    expected = {"index_vectors": index_vectors, "repo_credentials": "tok123"}
    assert permanent_pipeline["add"]["codegraph_config"] == expected
    assert permanent_pipeline["cognify"]["codegraph_config"] == expected
    for call in ("add", "cognify"):
        assert "index_vectors" not in permanent_pipeline[call]
        assert "repo_credentials" not in permanent_pipeline[call]


@pytest.mark.asyncio
async def test_explicit_codegraph_config_wins_over_top_level_spelling(permanent_pipeline):
    await remember_module.remember(
        "/some/repo",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        index_vectors=False,
        codegraph_config={"index_vectors": True},
    )

    assert permanent_pipeline["cognify"]["codegraph_config"] == {"index_vectors": True}


@pytest.mark.asyncio
async def test_unknown_codegraph_config_key_is_rejected(permanent_pipeline):
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: index_vector"):
        await remember_module.remember(
            "/some/repo",
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            codegraph_config={"index_vector": True},
        )

    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_session_id(permanent_pipeline):
    # The session path never runs cognify, so the option would be ignored.
    with pytest.raises(ValueError, match="only supported for normal ingestion"):
        await remember_module.remember(
            "note",
            session_id="s1",
            user=SimpleNamespace(id=uuid4()),
            index_vectors=True,
        )

    assert permanent_pipeline == {}


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
