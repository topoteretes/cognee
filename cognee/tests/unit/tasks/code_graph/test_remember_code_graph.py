"""remember(content_type='code') stores each repository through add() and builds it
through cognify's CODE_REPO route (SDK-783, SDK-793)."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.remember.remember import remember

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
code_repo_module = importlib.import_module("cognee.tasks.code_graph.code_repo")
migrations_module = importlib.import_module("cognee.modules.migrations.startup")
cognify_config_module = importlib.import_module("cognee.modules.cognify.config")


@pytest.fixture
def code_remember_env(monkeypatch, tmp_path):
    """Stub out migrations, dataset and repo resolution, the Data row, and its cognify run."""
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.setattr(migrations_module, "run_migrations_and_block", AsyncMock())

    dataset = SimpleNamespace(id=uuid4(), name="my_code")
    user = SimpleNamespace(id=uuid4())
    datasets_mock = AsyncMock(return_value=(user, [dataset]))
    monkeypatch.setattr(remember_module, "resolve_authorized_user_datasets", datasets_mock)

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    resolve_mock = AsyncMock(return_value=repo_dir)
    monkeypatch.setattr(resolve_module, "resolve_repo_source", resolve_mock)

    data_row = SimpleNamespace(id=uuid4())
    add_repo_mock = AsyncMock(return_value=data_row)
    monkeypatch.setattr(code_repo_module, "add_code_repository", add_repo_mock)

    cognify_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(code_repo_module, "cognify_code_repository", cognify_mock)

    return {
        "repo_dir": repo_dir,
        "resolve": resolve_mock,
        "cognify": cognify_mock,
        "dataset": dataset,
        "user": user,
        "datasets": datasets_mock,
        "data_row": data_row,
        "add_repo": add_repo_mock,
    }


def _run_info(status, dataset_name="my_code", error_message=None):
    return SimpleNamespace(
        status=status,
        pipeline_run_id=uuid4(),
        dataset_name=dataset_name,
        error_message=error_message,
    )


@pytest.mark.asyncio
async def test_single_repo_is_stored_then_built_through_the_cognify_route(code_remember_env):
    result = await remember(
        "https://github.com/org/repo",
        dataset_name="my_code",
        content_type="code",
    )

    code_remember_env["resolve"].assert_awaited_once_with(
        "https://github.com/org/repo", credentials=None
    )
    code_remember_env["add_repo"].assert_awaited_once_with(
        code_remember_env["repo_dir"],
        user=code_remember_env["user"],
        dataset=code_remember_env["dataset"],
        source_url="https://github.com/org/repo",
        index_vectors=False,
    )
    code_remember_env["cognify"].assert_awaited_once_with(
        code_remember_env["data_row"], code_remember_env["dataset"], code_remember_env["user"]
    )

    assert result.status == "completed"
    assert result.items_processed == 1
    assert result.items[0]["kind"] == "code_repository"
    assert result.items[0]["source"] == "https://github.com/org/repo"
    assert result.items[0]["id"] == str(code_remember_env["data_row"].id)
    assert result.dataset_id == str(code_remember_env["dataset"].id)


@pytest.mark.asyncio
async def test_repo_list_stores_and_builds_each_repo(code_remember_env):
    repos = ["https://github.com/org/a", "https://github.com/org/b", "/local/c"]

    result = await remember(repos, content_type="code")

    assert code_remember_env["resolve"].await_count == 3
    assert code_remember_env["add_repo"].await_count == 3
    assert code_remember_env["cognify"].await_count == 3
    assert [item["source"] for item in result.items] == repos
    # One dataset for the whole batch, resolved before the first repo runs.
    code_remember_env["datasets"].assert_awaited_once()


@pytest.mark.asyncio
async def test_local_repo_row_records_no_source_url(code_remember_env):
    await remember(str(code_remember_env["repo_dir"]), content_type="code")

    assert code_remember_env["add_repo"].await_args.kwargs["source_url"] is None


@pytest.mark.asyncio
async def test_index_vectors_is_recorded_on_the_row(code_remember_env):
    """The route reads the flag off the row, so remember() only has to store it."""
    await remember("/some/repo", content_type="code", index_vectors=True)

    assert code_remember_env["add_repo"].await_args.kwargs["index_vectors"] is True


@pytest.mark.asyncio
async def test_code_defaults_to_graph_only(code_remember_env):
    await remember("/some/repo", content_type="code")

    assert code_remember_env["add_repo"].await_args.kwargs["index_vectors"] is False


@pytest.mark.asyncio
async def test_no_other_builder_runs(code_remember_env, monkeypatch):
    """The row's cognify run is the only graph build: no custom pipeline, no
    stamp written by hand, no dataset lock taken by remember itself."""
    custom_pipeline_module = importlib.import_module("cognee.modules.run_custom_pipeline")
    data_methods_module = importlib.import_module("cognee.modules.data.methods")
    lock_module = importlib.import_module("cognee.infrastructure.locks.dataset_lock")
    custom = AsyncMock()
    stamp = AsyncMock()
    monkeypatch.setattr(custom_pipeline_module, "run_custom_pipeline", custom)
    monkeypatch.setattr(data_methods_module, "mark_data_processed", stamp)
    monkeypatch.setattr(
        lock_module,
        "dataset_lock",
        lambda *_a, **_k: pytest.fail("remember() must not take the dataset lock itself"),
    )

    result = await remember("/some/repo", content_type="code")

    assert result.status == "completed"
    custom.assert_not_awaited()
    stamp.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_id_is_rejected_for_code(code_remember_env):
    with pytest.raises(ValueError, match="session_id"):
        await remember("/some/repo", content_type="code", session_id="s1")


@pytest.mark.asyncio
async def test_non_string_data_is_rejected(code_remember_env):
    with pytest.raises(ValueError, match="repository path or git URL"):
        await remember([{"not": "a repo"}], content_type="code")


@pytest.mark.asyncio
async def test_index_vectors_without_code_content_type_is_rejected(code_remember_env, monkeypatch):
    # A text remember() runs the keyless extractor gate before argument
    # validation; disable the preflight so this test checks the argument only.
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    with pytest.raises(ValueError, match="index_vectors"):
        await remember("some text", index_vectors=True)


@pytest.mark.asyncio
async def test_blocking_captures_pipeline_run_info(code_remember_env):
    run_info = _run_info("PipelineRunCompleted")
    code_remember_env["cognify"].return_value = run_info

    result = await remember("/some/repo", dataset_name="my_code", content_type="code")

    assert result.status == "completed"
    assert result.dataset_id == str(code_remember_env["dataset"].id)
    assert result.pipeline_run_id == str(run_info.pipeline_run_id)
    assert result.items[0]["pipeline_run_id"] == str(run_info.pipeline_run_id)


@pytest.mark.asyncio
async def test_blocking_errored_run_marks_result_errored(code_remember_env):
    code_remember_env["cognify"].return_value = _run_info(
        "PipelineRunErrored", error_message="enola crashed"
    )

    result = await remember("/some/repo", content_type="code")

    assert result.status == "errored"
    assert result.error == "/some/repo: enola crashed"
    assert result.items[0]["status"] == "errored"
    assert result.items[0]["id"] == str(code_remember_env["data_row"].id)
    assert result.items_processed == 0


@pytest.mark.asyncio
async def test_run_in_background_returns_running_then_completes(code_remember_env):
    resolved_dataset = code_remember_env["dataset"]
    result = await remember(
        "https://github.com/org/repo",
        dataset_name="my_code",
        content_type="code",
        run_in_background=True,
    )

    assert result.status == "running"
    assert result.dataset_id == str(resolved_dataset.id)
    assert result.dataset_name == "my_code"

    await result

    assert result.status == "completed"
    assert result.items_processed == 1
    assert result.items[0]["id"] == str(code_remember_env["data_row"].id)
    code_remember_env["cognify"].assert_awaited_once_with(
        code_remember_env["data_row"], resolved_dataset, code_remember_env["user"]
    )


@pytest.mark.asyncio
async def test_background_failure_continues_batch(code_remember_env):
    repo_dir = code_remember_env["repo_dir"]
    code_remember_env["resolve"].side_effect = [RuntimeError("clone failed"), repo_dir]

    result = await remember(
        ["https://github.com/org/bad", "https://github.com/org/good"],
        content_type="code",
        run_in_background=True,
    )
    await result

    assert result.status == "errored"
    assert "clone failed" in result.error
    assert [item["source"] for item in result.items] == [
        "https://github.com/org/bad",
        "https://github.com/org/good",
    ]
    assert result.items[0]["status"] == "errored"
    assert "id" not in result.items[0]
    assert result.items[1]["id"] == str(code_remember_env["data_row"].id)
    assert result.items_processed == 1
    assert code_remember_env["cognify"].await_count == 1


@pytest.mark.asyncio
async def test_failure_after_the_row_is_stored_keeps_its_id(code_remember_env):
    """A repo whose build raises after its Data row was written still reports
    that row's id, so the caller can forget it."""
    code_remember_env["cognify"].side_effect = RuntimeError("enola crashed")

    result = await remember("/some/repo", content_type="code", run_in_background=True)
    await result

    assert result.items[0]["status"] == "errored"
    assert result.items[0]["id"] == str(code_remember_env["data_row"].id)


@pytest.mark.asyncio
async def test_result_error_names_every_failed_repo(code_remember_env):
    """A pipeline-errored repo and a raising repo in one batch both reach
    result.error, each with its own cause."""
    repo_dir = code_remember_env["repo_dir"]
    code_remember_env["resolve"].side_effect = [repo_dir, RuntimeError("clone failed")]
    code_remember_env["cognify"].return_value = _run_info(
        "PipelineRunErrored", error_message="enola crashed"
    )

    result = await remember(
        ["https://github.com/org/a", "https://github.com/org/b"],
        content_type="code",
        raise_on_error=False,
    )

    assert result.status == "errored"
    assert result.items[0]["error"] == "enola crashed"
    assert result.error == (
        "https://github.com/org/a: enola crashed; https://github.com/org/b: clone failed"
    )


@pytest.mark.asyncio
async def test_blocking_raises_a_failed_repo_by_default(code_remember_env):
    code_remember_env["resolve"].side_effect = RuntimeError("clone failed")

    with pytest.raises(RuntimeError, match="clone failed"):
        await remember("https://github.com/org/bad", content_type="code")


@pytest.mark.asyncio
async def test_blocking_with_raise_on_error_false_reports_failed_repos(code_remember_env):
    """The HTTP router passes raise_on_error=False and answers an errored result
    with 409; a failed repo must become an errored item, not an exception, and
    must not abort the rest of the batch."""
    repo_dir = code_remember_env["repo_dir"]
    code_remember_env["resolve"].side_effect = [RuntimeError("clone failed"), repo_dir]

    result = await remember(
        ["https://github.com/org/bad", "https://github.com/org/good"],
        content_type="code",
        raise_on_error=False,
    )

    assert result.status == "errored"
    assert "clone failed" in result.error
    assert result.items[0]["status"] == "errored"
    assert result.items[1]["id"] == str(code_remember_env["data_row"].id)
    assert result.items_processed == 1


@pytest.mark.asyncio
async def test_blocking_ignores_background_machinery(code_remember_env):
    result = await remember("/some/repo", content_type="code")

    assert result._task is None
    assert result.done


@pytest.mark.asyncio
async def test_code_route_never_resolves_the_graph_extractor(code_remember_env, monkeypatch):
    """A keyless install without gliner2 must still build code graphs.

    The code route runs enola only, so the extractor gate that a keyless
    text remember() hits (KeylessExtractorNotInstalledError) must not run.
    """

    def _gate_would_fail(*_args, **_kwargs):
        raise cognify_config_module.KeylessExtractorNotInstalledError()

    monkeypatch.setattr(cognify_config_module, "resolve_extractor", _gate_would_fail)

    result = await remember(
        str(code_remember_env["repo_dir"]), dataset_name="my_code", content_type="code"
    )

    assert result.status == "completed"
    code_remember_env["cognify"].assert_awaited_once()
