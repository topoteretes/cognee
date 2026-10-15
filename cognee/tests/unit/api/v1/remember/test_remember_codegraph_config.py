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


def _remember(data, **kwargs):
    """remember() with a throwaway dataset and user, and no automatic improve."""
    if "session_id" not in kwargs:
        kwargs.setdefault("dataset_id", uuid4())
    kwargs.setdefault("user", SimpleNamespace(id=uuid4()))
    kwargs.setdefault("self_improvement", False)
    return remember_module.remember(data, **kwargs)


@pytest.mark.asyncio
async def test_codegraph_config_reaches_add_and_cognify(permanent_pipeline):
    config = {"index_vectors": True, "repo_credentials": "tok123"}

    result = await _remember("https://github.com/acme/api", codegraph_config=config)

    assert result.status == "completed"
    assert permanent_pipeline["add"]["codegraph_config"] == config
    assert permanent_pipeline["cognify"]["codegraph_config"] == config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"index_vectors": False, "repo_credentials": "tok123"},
            {"index_vectors": False, "repo_credentials": "tok123"},
        ),
        (
            {"index_vectors": True, "repo_credentials": "tok123"},
            {"index_vectors": True, "repo_credentials": "tok123"},
        ),
        # An explicit codegraph_config wins over the top-level spelling.
        (
            {"index_vectors": False, "codegraph_config": {"index_vectors": True}},
            {"index_vectors": True},
        ),
    ],
)
async def test_top_level_spellings_fold_into_codegraph_config(permanent_pipeline, kwargs, expected):
    result = await _remember("https://github.com/acme/api", **kwargs)

    assert result.status == "completed"
    assert permanent_pipeline["add"]["codegraph_config"] == expected
    assert permanent_pipeline["cognify"]["codegraph_config"] == expected
    for call in ("add", "cognify"):
        assert "index_vectors" not in permanent_pipeline[call]
        assert "repo_credentials" not in permanent_pipeline[call]


@pytest.mark.asyncio
async def test_unknown_codegraph_config_key_is_rejected(permanent_pipeline):
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: index_vector"):
        await _remember("/some/repo", codegraph_config={"index_vector": True})

    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_session_id(permanent_pipeline):
    # The session path never runs cognify, so the option would be ignored.
    with pytest.raises(ValueError, match="only supported for normal ingestion"):
        await _remember("note", session_id="s1", index_vectors=True)

    assert permanent_pipeline == {}


@pytest.fixture
def improve_calls(monkeypatch):
    """Record improve() calls, with automatic improve enabled."""
    calls = []

    async def fake_improve(**kwargs):
        calls.append(kwargs)

    # The package re-exports improve() under its own module's name, so patch
    # the package attribute remember() imports.
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.improve"), "improve", fake_improve)
    monkeypatch.setenv("IMPROVE_AUTO_ENABLED", "true")
    return calls


@pytest.mark.asyncio
async def test_removed_code_content_type_is_ignored(permanent_pipeline, improve_calls, caplog):
    # Accepted so existing callers do not break, and ignored: the call is a
    # plain remember(data) -- no translation into codegraph_config, and the
    # plain call's improve default -- with a warning naming the replacement.
    import logging
    import warnings

    dataset_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    await remember_module.remember(
        "https://github.com/acme/api", dataset_id=dataset_id, user=user, repo_credentials="tok123"
    )
    plain = dict(permanent_pipeline)
    permanent_pipeline.clear()
    assert len(improve_calls) == 1

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        result = await remember_module.remember(
            "https://github.com/acme/api",
            dataset_id=dataset_id,
            user=user,
            repo_credentials="tok123",
            content_type="code",
        )

    assert result.status == "completed"
    assert permanent_pipeline == plain
    assert "content_type" not in permanent_pipeline["add"]
    assert "content_type" not in permanent_pipeline["cognify"]
    deprecations = [w for w in caught if "content_type" in str(w.message)]
    assert [w.category for w in deprecations] == [FutureWarning]
    assert deprecations[0].filename == __file__
    assert '"treat_as_repository": True' in str(deprecations[0].message)
    assert 'content_type="code"' in caplog.text
    assert len(improve_calls) == 2


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_for_memory_entries(permanent_pipeline):
    from cognee.memory import QAEntry

    with pytest.raises(ValueError, match="not supported for memory entries"):
        await _remember(
            QAEntry(question="q", answer="a"),
            session_id="s1",
            codegraph_config={"treat_as_repository": True},
        )


@pytest.mark.asyncio
async def test_mistyped_codegraph_values_are_rejected_before_any_work(permanent_pipeline):
    with pytest.raises(TypeError, match="must be a bool"):
        await _remember(
            "https://github.com/acme/api", codegraph_config={"include_documents": "false"}
        )

    assert permanent_pipeline == {}
