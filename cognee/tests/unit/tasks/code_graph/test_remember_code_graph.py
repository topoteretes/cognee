"""remember(content_type='code') routes repos through the code-graph pipeline."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.remember.remember import remember

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
pipeline_module = importlib.import_module("cognee.modules.run_custom_pipeline")
migrations_module = importlib.import_module("cognee.modules.migrations.startup")


@pytest.fixture
def code_remember_env(monkeypatch, tmp_path):
    """Stub out migrations, repo resolution, and the pipeline run."""
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.setattr(migrations_module, "run_migrations_and_block", AsyncMock())

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    resolve_mock = AsyncMock(return_value=repo_dir)
    monkeypatch.setattr(resolve_module, "resolve_repo_source", resolve_mock)

    pipeline_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(pipeline_module, "run_custom_pipeline", pipeline_mock)

    return {"repo_dir": repo_dir, "resolve": resolve_mock, "pipeline": pipeline_mock}


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
    assert call.kwargs["dataset"] == "my_code"
    assert call.kwargs["pipeline_name"] == "code_graph_pipeline"
    assert call.kwargs["data"] == str(code_remember_env["repo_dir"])
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
    assert code_remember_env["pipeline"].await_count == 3
    assert [item["source"] for item in result.items] == repos


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
async def test_index_vectors_without_code_content_type_is_rejected(code_remember_env):
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


@pytest.fixture
def resolved_dataset(monkeypatch):
    """Stub upfront dataset resolution used by the background code branch."""
    dataset = SimpleNamespace(id=uuid4(), name="my_code")
    user = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(
        remember_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(user, [dataset])),
    )
    return dataset


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
async def test_run_in_background_returns_running_then_completes(
    code_remember_env, resolved_dataset
):
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
    assert code_remember_env["pipeline"].await_args.kwargs["dataset"] == resolved_dataset.id


@pytest.mark.asyncio
async def test_background_failure_continues_batch(code_remember_env, resolved_dataset):
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
async def test_blocking_ignores_background_machinery(code_remember_env):
    result = await remember("/some/repo", content_type="code")

    assert result._task is None
    assert result.done
