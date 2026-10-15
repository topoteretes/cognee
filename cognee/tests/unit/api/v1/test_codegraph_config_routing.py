"""codegraph_config through add(), cognify() and remember() (SDK-793).

remember(content_type="code") is accepted and ignored: a repository takes the
same add() + cognify() path as any other input, and the old code-only kwargs
fold into codegraph_config.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as serve_state

add_module = importlib.import_module("cognee.api.v1.add.add")
cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
code_repo_module = importlib.import_module("cognee.tasks.code_graph.code_repo")
extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")

_REPO = "https://github.com/org/repo"
_USER_ID = uuid4()
_DATASET_ID = uuid4()


@pytest.fixture
def executor(monkeypatch):
    """add() with the database and pipeline stubbed; the returned mock records the run."""
    dataset = SimpleNamespace(id=uuid4(), name="code")
    user = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "validate_provider_config", Mock()
    )
    monkeypatch.setattr(add_module, "setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    monkeypatch.setattr(
        add_module, "resolve_authorized_user_dataset", AsyncMock(return_value=(user, dataset))
    )
    monkeypatch.setattr(
        add_module, "resolve_dlt_sources", AsyncMock(side_effect=lambda data, **kw: (data, None))
    )
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", AsyncMock())
    run = AsyncMock(return_value=None)
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: run)
    return run


# --- add() -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_hands_the_repository_options_to_both_resolve_passes(executor, tmp_path):
    config = {"treat_as_repository": True, "include_documents": False, "repo_credentials": "t"}

    await add_module.add([str(tmp_path), _REPO], codegraph_config=config)

    run = executor.await_args
    # run_tasks resolves items before the task list runs, from the extras ...
    assert run.kwargs["extras"]["codegraph_config"] == config
    # ... and the add task resolves them again.
    task_kwargs = run.kwargs["tasks"][0].default_params["kwargs"]
    assert task_kwargs["credentials"] == "t"
    assert task_kwargs["include_documents"] is False
    assert task_kwargs["treat_as_repository"] is True


@pytest.mark.asyncio
async def test_add_defaults_keep_detection_and_documents(executor):
    await add_module.add(_REPO)

    task_kwargs = executor.await_args.kwargs["tasks"][0].default_params["kwargs"]
    assert task_kwargs["credentials"] is None
    assert task_kwargs["include_documents"] is True
    assert task_kwargs["treat_as_repository"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item, match",
    [
        ("just some text", "is not a repository"),
        ("/no/such/repository", "is not a repository"),
        (b"bytes", "expects repository paths or git URLs"),
    ],
)
async def test_treat_as_repository_refuses_other_items_before_writing(executor, item, match):
    with pytest.raises(ValueError, match=match):
        await add_module.add(item, codegraph_config={"treat_as_repository": True})
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_treat_as_repository_refuses_a_file(executor, tmp_path):
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n")

    with pytest.raises(ValueError, match="is not a repository"):
        await add_module.add(str(document), codegraph_config={"treat_as_repository": True})


@pytest.mark.asyncio
async def test_treat_as_repository_accepts_directories_and_remotes(executor, tmp_path):
    await add_module.add(
        [str(tmp_path), tmp_path, "git@host:org/repo.git", "https://example.com/team/repo"],
        codegraph_config={"treat_as_repository": True},
    )
    executor.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_refuses_an_invalid_config(executor):
    with pytest.raises(TypeError, match="must be a bool"):
        await add_module.add(_REPO, codegraph_config={"include_documents": "no"})
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_refuses_repository_options_on_a_remote_instance(monkeypatch):
    client = Mock(add=AsyncMock(return_value={}))
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    for config in (
        {"repo_credentials": "t"},
        {"treat_as_repository": True},
        {"include_documents": False},
    ):
        with pytest.raises(ValueError, match="remote Cognee instance"):
            await add_module.add(_REPO, codegraph_config=config)
    client.add.assert_not_awaited()

    # Defaults change nothing, so they still go through.
    await add_module.add(_REPO, codegraph_config={"include_documents": True})
    client.add.assert_awaited_once()


# --- cognify() ----------------------------------------------------------------


def test_the_code_repo_route_embeds_only_when_asked():
    (graph_only_task,) = code_repo_module.get_code_repo_tasks()
    (embedding_task,) = code_repo_module.get_code_repo_tasks(index_vectors=True)

    assert graph_only_task.default_params["kwargs"]["index_vectors"] is False
    assert embedding_task.default_params["kwargs"]["index_vectors"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("index_vectors", [False, True])
async def test_extract_code_repo_graph_writes_vectors_only_when_asked(
    tmp_path, monkeypatch, index_vectors
):
    import json

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo_path": str(tmp_path)}))
    item = SimpleNamespace(
        id=uuid4(),
        raw_data_location=str(manifest),
        system_metadata={"source": "code_repo"},
        external_metadata=None,
    )
    write = AsyncMock(return_value="state")
    monkeypatch.setattr(extract_module, "extract_code_graph", AsyncMock(return_value=[]))
    monkeypatch.setattr(extract_module, "add_code_graph_data_points", write)
    monkeypatch.setattr(extract_module, "add_code_graph_edges", AsyncMock())

    await code_repo_module.extract_code_repo_graph([item], index_vectors=index_vectors)

    assert write.await_args.kwargs["graph_only"] is (not index_vectors)


@pytest.mark.asyncio
@patch.object(serve_state, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[])
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
@patch.object(cognify_module, "get_code_repo_tasks", return_value=[])
async def test_cognify_passes_index_vectors_to_the_code_repo_route(
    mock_code_repo_tasks, mock_resolver, mock_migrations, mock_default_tasks, mock_executor, _
):
    mock_executor.return_value = AsyncMock(return_value={})

    await cognify_module.cognify(codegraph_config={"index_vectors": True})
    await cognify_module.cognify()

    assert [c.kwargs for c in mock_code_repo_tasks.call_args_list] == [
        {"index_vectors": True},
        {"index_vectors": False},
    ]


@pytest.mark.asyncio
async def test_cognify_refuses_index_vectors_on_a_remote_instance(monkeypatch):
    client = Mock(cognify=AsyncMock(return_value={}))
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    with pytest.raises(ValueError, match="remote Cognee instance"):
        await cognify_module.cognify(codegraph_config={"index_vectors": True})
    client.cognify.assert_not_awaited()


# --- remember() ---------------------------------------------------------------


def test_remember_sends_codegraph_config_to_add_and_cognify():
    assert "codegraph_config" in remember_module._SHARED


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"content_type": "code"}, {}),
        ({"content_type": "code", "index_vectors": None, "repo_credentials": None}, {}),
        (
            {"content_type": "code", "index_vectors": True, "repo_credentials": "t"},
            {"codegraph_config": {"index_vectors": True, "repo_credentials": "t"}},
        ),
        (
            {"index_vectors": True, "codegraph_config": {"include_documents": False}},
            {"codegraph_config": {"index_vectors": True, "include_documents": False}},
        ),
        ({"codegraph_config": {"index_vectors": None}}, {}),
        ({"content_type": "skills"}, {"content_type": "skills"}),
    ],
)
def test_code_options_are_normalized(kwargs, expected):
    remember_module._normalize_code_options("data", kwargs)
    assert kwargs == expected


def test_a_folded_kwarg_must_agree_with_the_config():
    kwargs = {"index_vectors": True, "codegraph_config": {"index_vectors": False}}
    with pytest.raises(ValueError, match="both as a keyword argument and in codegraph_config"):
        remember_module._normalize_code_options("data", kwargs)


def test_a_memory_source_keeps_its_index_vectors():
    from cognee.modules.migration.sources.base import MemorySource

    source = MagicMock(spec=MemorySource)
    kwargs = {"content_type": "code", "index_vectors": False}
    remember_module._normalize_code_options(source, kwargs)
    assert kwargs == {"index_vectors": False}


async def _remember_calls(**kwargs):
    """Run remember() with add/cognify mocked; return their call kwargs."""
    user = MagicMock(id=_USER_ID, tenant_id=None)
    dataset = MagicMock(id=_DATASET_ID)
    with (
        patch("cognee.shared.utils.send_telemetry"),
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch("cognee.modules.engine.operations.setup.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=user)),
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock),
        patch.object(
            remember_module,
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(user, [dataset])),
        ),
        patch("cognee.api.v1.add.add", AsyncMock()) as add,
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})) as cognify,
    ):
        await remember_module.remember(_REPO, dataset_name="code", self_improvement=False, **kwargs)
    return add.await_args, cognify.await_args


@pytest.mark.asyncio
async def test_remember_with_content_type_code_is_plain_remember():
    plain_add, plain_cognify = await _remember_calls()
    code_add, code_cognify = await _remember_calls(content_type="code")

    def without_user(call):
        return call.args, {key: value for key, value in call.kwargs.items() if key != "user"}

    assert without_user(code_add) == without_user(plain_add)
    assert without_user(code_cognify) == without_user(plain_cognify)
    assert "content_type" not in code_add.kwargs


@pytest.mark.asyncio
async def test_remember_folds_the_legacy_kwargs_for_add_and_cognify():
    add, cognify = await _remember_calls(
        content_type="code", index_vectors=True, repo_credentials="t"
    )

    expected = {"index_vectors": True, "repo_credentials": "t"}
    assert add.kwargs["codegraph_config"] == expected
    assert cognify.kwargs["codegraph_config"] == expected


@pytest.mark.asyncio
async def test_remember_refuses_codegraph_config_where_it_would_be_dropped(monkeypatch):
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    config = {"include_documents": False}

    with pytest.raises(ValueError, match="codegraph_config is not supported when session_id"):
        await remember_module.remember(_REPO, session_id="s1", codegraph_config=config)
    with pytest.raises(ValueError, match="codegraph_config is supported only without content"):
        await remember_module.remember(_REPO, content_type="skills", codegraph_config=config)


def test_auto_presort_skips_a_folder_the_caller_calls_a_repository(tmp_path, monkeypatch):
    monkeypatch.setenv(remember_module.PRESORT_FOLDERS_ENV, "true")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)

    kwargs = {"codegraph_config": {"treat_as_repository": True}}
    assert not remember_module._should_auto_presort(
        str(tmp_path), "main_dataset", None, None, kwargs
    )
