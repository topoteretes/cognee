"""remember(content_type='code') routes repos through the code-graph pipeline."""

import importlib
from unittest.mock import AsyncMock

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

    code_remember_env["resolve"].assert_awaited_once_with("https://github.com/org/repo")
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
