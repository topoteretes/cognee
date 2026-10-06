"""remember(content_type='code') routes repos through the code-graph pipeline."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.remember.remember import remember

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
code_repo_module = importlib.import_module("cognee.tasks.code_graph.code_repo")
data_methods_module = importlib.import_module("cognee.modules.data.methods")
pipeline_module = importlib.import_module("cognee.modules.run_custom_pipeline")
migrations_module = importlib.import_module("cognee.modules.migrations.startup")
cognify_config_module = importlib.import_module("cognee.modules.cognify.config")


@pytest.fixture
def code_remember_env(monkeypatch, tmp_path):
    """Stub out migrations, dataset and repo resolution, the Data row, and the pipeline run."""
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

    mark_processed_mock = AsyncMock()
    monkeypatch.setattr(data_methods_module, "mark_data_processed", mark_processed_mock)

    pipeline_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(pipeline_module, "run_custom_pipeline", pipeline_mock)

    return {
        "repo_dir": repo_dir,
        "resolve": resolve_mock,
        "pipeline": pipeline_mock,
        "dataset": dataset,
        "user": user,
        "datasets": datasets_mock,
        "data_row": data_row,
        "add_repo": add_repo_mock,
        "mark_processed": mark_processed_mock,
    }


@pytest.mark.asyncio
async def test_single_repo_runs_code_graph_pipeline(code_remember_env):
    result = await remember(
        "https://github.com/org/repo",
        dataset_name="my_code",
        content_type="code",
    )

    code_remember_env["resolve"].assert_awaited_once_with(
        "https://github.com/org/repo", credentials=None
    )
    code_remember_env["pipeline"].assert_awaited_once()
    call = code_remember_env["pipeline"].await_args
    assert call.kwargs["dataset"] == code_remember_env["dataset"].id
    assert call.kwargs["pipeline_name"] == "code_graph_pipeline"
    assert call.kwargs["data"] == [code_remember_env["data_row"]]
    assert len(call.kwargs["tasks"]) == 3

    assert result.status == "completed"
    assert result.items_processed == 1
    assert result.items[0]["kind"] == "code_repository"
    assert result.items[0]["source"] == "https://github.com/org/repo"


@pytest.mark.asyncio
async def test_repo_list_runs_pipeline_per_repo(code_remember_env):
    repos = ["https://github.com/org/a", "https://github.com/org/b", "/local/c"]

    result = await remember(repos, content_type="code")

    assert code_remember_env["resolve"].await_count == 3
    assert code_remember_env["add_repo"].await_count == 3
    assert code_remember_env["pipeline"].await_count == 3
    assert [item["source"] for item in result.items] == repos
    # One dataset for the whole batch, resolved before the first repo runs.
    code_remember_env["datasets"].assert_awaited_once()


@pytest.mark.asyncio
async def test_each_repo_is_stored_as_a_data_row_in_the_dataset(code_remember_env):
    """SDK-783: a repository gets a Data row, so the caller has a data_id to forget."""
    result = await remember(
        "https://github.com/org/repo", dataset_name="my_code", content_type="code"
    )

    code_remember_env["add_repo"].assert_awaited_once_with(
        code_remember_env["repo_dir"],
        user=code_remember_env["user"],
        dataset=code_remember_env["dataset"],
        source_url="https://github.com/org/repo",
        skip_connection_test=True,
    )
    assert result.items[0]["id"] == str(code_remember_env["data_row"].id)
    assert result.dataset_id == str(code_remember_env["dataset"].id)


@pytest.mark.asyncio
async def test_local_repo_row_records_no_source_url(code_remember_env):
    await remember(str(code_remember_env["repo_dir"]), content_type="code")

    assert code_remember_env["add_repo"].await_args.kwargs["source_url"] is None


@pytest.mark.asyncio
async def test_completed_repo_is_stamped_as_cognified(code_remember_env):
    """A later cognify() of the dataset must not rerun enola for a repo already built."""
    await remember("/some/repo", content_type="code")

    code_remember_env["mark_processed"].assert_awaited_once_with(
        code_remember_env["data_row"].id,
        code_remember_env["dataset"].id,
        pipeline_names=("cognify_pipeline", "code_graph_pipeline"),
    )


@pytest.mark.asyncio
async def test_errored_repo_is_not_stamped_as_cognified(code_remember_env):
    pipeline_result, _, _ = _pipeline_result("PipelineRunErrored")
    code_remember_env["pipeline"].return_value = pipeline_result

    await remember("/some/repo", content_type="code")

    code_remember_env["mark_processed"].assert_not_awaited()


@pytest.mark.asyncio
async def test_index_vectors_is_forwarded_to_tasks(code_remember_env):
    await remember("/some/repo", content_type="code", index_vectors=True)

    tasks = code_remember_env["pipeline"].await_args.kwargs["tasks"]
    assert tasks[1].default_params["kwargs"]["graph_only"] is False


@pytest.mark.asyncio
async def test_code_defaults_to_graph_only(code_remember_env):
    await remember("/some/repo", content_type="code")

    tasks = code_remember_env["pipeline"].await_args.kwargs["tasks"]
    assert tasks[1].default_params["kwargs"]["graph_only"] is True


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


def _pipeline_result(status, dataset_name="my_code"):
    """Shape of a blocking run_custom_pipeline return: {dataset_id: run_info}."""
    dataset_uuid = uuid4()
    run_id = uuid4()
    return (
        {
            dataset_uuid: SimpleNamespace(
                status=status, pipeline_run_id=run_id, dataset_name=dataset_name
            )
        },
        dataset_uuid,
        run_id,
    )


@pytest.mark.asyncio
async def test_blocking_captures_pipeline_run_info(code_remember_env):
    pipeline_result, dataset_uuid, run_id = _pipeline_result("PipelineRunCompleted")
    code_remember_env["pipeline"].return_value = pipeline_result

    result = await remember("/some/repo", dataset_name="my_code", content_type="code")

    assert result.status == "completed"
    assert result.dataset_id == str(dataset_uuid)
    assert result.pipeline_run_id == str(run_id)
    assert result.items[0]["pipeline_run_id"] == str(run_id)


@pytest.mark.asyncio
async def test_blocking_errored_run_marks_result_errored(code_remember_env):
    pipeline_result, _, _ = _pipeline_result("PipelineRunErrored")
    code_remember_env["pipeline"].return_value = pipeline_result

    result = await remember("/some/repo", content_type="code")

    assert result.status == "errored"
    assert "code_graph_pipeline errored" in result.error
    assert result.items[0]["status"] == "errored"
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
    assert code_remember_env["pipeline"].await_args.kwargs["dataset"] == resolved_dataset.id


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
    assert result.items[1].get("status") != "errored"
    assert result.items_processed == 1
    # The good repo still ran through the pipeline.
    assert code_remember_env["pipeline"].await_count == 1


@pytest.mark.asyncio
async def test_failure_after_the_row_is_stored_keeps_its_id(code_remember_env):
    """A repo whose pipeline raises after its Data row was written still reports
    that row's id, so the caller can forget it."""
    code_remember_env["pipeline"].side_effect = RuntimeError("enola crashed")

    result = await remember("/some/repo", content_type="code", run_in_background=True)
    await result

    assert result.items[0]["status"] == "errored"
    assert result.items[0]["id"] == str(code_remember_env["data_row"].id)
    code_remember_env["mark_processed"].assert_not_awaited()


@pytest.mark.asyncio
async def test_result_error_names_every_failed_repo(code_remember_env):
    """A pipeline-errored repo and a raising repo in one batch both reach
    result.error, each with its own cause."""
    repo_dir = code_remember_env["repo_dir"]
    code_remember_env["resolve"].side_effect = [repo_dir, RuntimeError("clone failed")]
    dataset_uuid = uuid4()
    code_remember_env["pipeline"].return_value = {
        dataset_uuid: SimpleNamespace(
            status="PipelineRunErrored",
            pipeline_run_id=uuid4(),
            dataset_name="my_code",
            error_message="enola crashed",
        )
    }

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
async def test_row_write_run_and_stamp_hold_the_dataset_lock(code_remember_env, monkeypatch):
    """The row's stamps are read-modify-written as a whole, so storing the row,
    building its graph, and stamping it must all happen under the dataset lock."""
    from contextlib import asynccontextmanager

    lock_module = importlib.import_module("cognee.infrastructure.locks.dataset_lock")
    events = []

    @asynccontextmanager
    async def _recording_lock(dataset_id):
        events.append(("lock", dataset_id))
        yield
        events.append(("unlock", dataset_id))

    monkeypatch.setattr(lock_module, "dataset_lock", _recording_lock)
    for name in ("add_repo", "pipeline", "mark_processed"):
        code_remember_env[name].side_effect = lambda *_args, _name=name, **_kwargs: events.append(
            (_name, None)
        )
    code_remember_env["add_repo"].side_effect = lambda *_a, **_k: (
        events.append(("add_repo", None)) or code_remember_env["data_row"]
    )

    await remember("/some/repo", content_type="code")

    dataset_id = code_remember_env["dataset"].id
    assert events == [
        ("lock", dataset_id),
        ("add_repo", None),
        ("pipeline", None),
        ("mark_processed", None),
        ("unlock", dataset_id),
    ]


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
    code_remember_env["pipeline"].assert_awaited_once()
