"""``remember(codegraph_config=...)`` reaches both add() and cognify(), and code needs no
content type.

Code repositories go through ``add()`` + ``cognify()`` like any other input
(SDK-793). ``codegraph_config`` is handed to both: add() reads
``repo_credentials``, ``include_documents`` and ``treat_as_repository``,
cognify() reads ``index_vectors`` for the CODE / CODE_REPO task lists. The
top-level ``index_vectors=`` / ``repo_credentials=`` spellings fold into it
with a FutureWarning, and the deprecated ``content_type="code"`` folds into it with
one.
"""

import importlib
import logging
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
preflight_module = importlib.import_module("cognee.modules.preflight")
cognify_config_module = importlib.import_module("cognee.modules.cognify.config")


@pytest.fixture(autouse=True)
def _no_db_setup(monkeypatch):
    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )


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
        # The same value both ways is one value.
        (
            {"index_vectors": True, "codegraph_config": {"index_vectors": True}},
            {"index_vectors": True},
        ),
    ],
)
async def test_top_level_spellings_fold_into_codegraph_config(permanent_pipeline, kwargs, expected):
    with pytest.warns(FutureWarning, match="codegraph_config=") as caught:
        result = await _remember("https://github.com/acme/api", **kwargs)

    assert result.status == "completed"
    assert permanent_pipeline["add"]["codegraph_config"] == expected
    assert permanent_pipeline["cognify"]["codegraph_config"] == expected
    for call in ("add", "cognify"):
        assert "index_vectors" not in permanent_pipeline[call]
        assert "repo_credentials" not in permanent_pipeline[call]
    # The warning points at the caller, not at remember()'s own frames.
    assert [w.filename for w in caught if "codegraph_config=" in str(w.message)] == [__file__]


@pytest.mark.asyncio
async def test_a_key_given_both_ways_with_different_values_raises(permanent_pipeline):
    """Neither spelling silently wins: index_vectors=True next to
    codegraph_config={"index_vectors": False} is a caller mistake."""
    with (
        pytest.warns(FutureWarning, match="codegraph_config="),
        pytest.raises(ValueError, match="index_vectors given both"),
    ):
        await _remember(
            "https://github.com/acme/api",
            index_vectors=True,
            codegraph_config={"index_vectors": False},
        )
    assert "add" not in permanent_pipeline


@pytest.mark.asyncio
async def test_unknown_codegraph_config_key_is_rejected(permanent_pipeline):
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: index_vector"):
        await _remember("/some/repo", codegraph_config={"index_vector": True})

    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_session_id(permanent_pipeline):
    # The session path never runs cognify, so the option would be ignored.
    with pytest.raises(ValueError, match="only supported for normal ingestion"):
        await _remember("note", session_id="s1", codegraph_config={"index_vectors": True})

    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_the_skills_content_type(permanent_pipeline):
    with pytest.raises(ValueError, match="only supported for normal ingestion"):
        await _remember(
            "/skills", content_type="skills", codegraph_config={"treat_as_repository": True}
        )

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
async def test_code_content_type_folds_into_codegraph_config(permanent_pipeline, caplog):
    """The deprecated spelling keeps its semantics through the one normal path:
    every item a declared repository, documents left out, forwarded to add()
    and cognify() exactly as a caller passing the dict would, with the
    FutureWarning naming that dict."""
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        result = await _remember("https://github.com/acme/api", content_type="code")

    assert result.status == "completed"
    folded = {"treat_as_repository": True, "include_documents": False}
    assert permanent_pipeline["add"]["codegraph_config"] == folded
    assert permanent_pipeline["cognify"]["codegraph_config"] == folded
    assert "content_type" not in permanent_pipeline["add"]
    assert "content_type" not in permanent_pipeline["cognify"]
    deprecations = [w for w in caught if "content_type" in str(w.message)]
    assert [w.category for w in deprecations] == [FutureWarning]
    assert deprecations[0].filename == __file__
    assert '"treat_as_repository": True' in str(deprecations[0].message)
    assert 'content_type="code"' in caplog.text


@pytest.mark.asyncio
async def test_code_content_type_merges_with_an_agreeing_config_and_refuses_a_conflicting_one(
    permanent_pipeline,
):
    with pytest.warns(FutureWarning, match="content_type"):
        await _remember(
            "https://github.com/acme/api",
            content_type="code",
            codegraph_config={"index_vectors": True, "include_documents": False},
        )
    assert permanent_pipeline["cognify"]["codegraph_config"] == {
        "treat_as_repository": True,
        "include_documents": False,
        "index_vectors": True,
    }
    permanent_pipeline.clear()

    with (
        pytest.warns(FutureWarning, match="content_type"),
        pytest.raises(ValueError, match="include_documents given both"),
    ):
        await _remember(
            "https://github.com/acme/api",
            content_type="code",
            codegraph_config={"include_documents": True},
        )
    assert "add" not in permanent_pipeline


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


@pytest.mark.asyncio
async def test_a_declared_repository_call_needs_no_llm_and_resolves_no_extractor(
    permanent_pipeline, monkeypatch
):
    """treat_as_repository: every item is a repository, cognify runs enola only."""
    validate = Mock()
    resolve = Mock(side_effect=AssertionError("the extractor must not be resolved"))
    monkeypatch.setattr(preflight_module, "validate_provider_config", validate)
    monkeypatch.setattr(cognify_config_module, "resolve_extractor", resolve)

    await _remember("git@github.com:acme/api.git", codegraph_config={"treat_as_repository": True})

    validate.assert_called_once_with(needs_llm=False)
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_a_plain_call_still_gates_on_the_resolved_extractor(permanent_pipeline, monkeypatch):
    validate = Mock()
    monkeypatch.setattr(preflight_module, "validate_provider_config", validate)
    monkeypatch.setattr(cognify_config_module, "resolve_extractor", Mock(return_value="llm"))

    # A note is a document call, whatever the config says about repositories.
    await _remember("a plain note", codegraph_config={"include_documents": False})

    validate.assert_called_once_with(needs_llm=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    ["https://github.com/acme/api", ["https://gitlab.com/group/api", "https://github.com/o/r"]],
)
async def test_detected_repository_urls_need_no_llm_either(permanent_pipeline, monkeypatch, data):
    """The preflight reads what the items are, not only the treat_as_repository flag."""
    validate = Mock()
    resolve = Mock(side_effect=AssertionError("the extractor must not be resolved"))
    monkeypatch.setattr(preflight_module, "validate_provider_config", validate)
    monkeypatch.setattr(cognify_config_module, "resolve_extractor", resolve)

    await _remember(data)

    validate.assert_called_once_with(needs_llm=False)


@pytest.mark.asyncio
async def test_a_detected_project_directory_needs_no_llm(permanent_pipeline, monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
    validate = Mock()
    monkeypatch.setattr(preflight_module, "validate_provider_config", validate)
    monkeypatch.setattr(
        cognify_config_module, "resolve_extractor", Mock(side_effect=AssertionError("resolved"))
    )

    await _remember(str(tmp_path))

    validate.assert_called_once_with(needs_llm=False)

    # Without a project marker the directory is documents: the extractor gates.
    (tmp_path / "pyproject.toml").unlink()
    monkeypatch.setattr(cognify_config_module, "resolve_extractor", Mock(return_value="llm"))
    await _remember(str(tmp_path))
    assert validate.call_args.kwargs == {"needs_llm": True}
