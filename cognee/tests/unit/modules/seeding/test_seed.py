import os
import time
from pathlib import Path

import pytest

import importlib

from cognee.modules.seeding.discovery import claude_code_project_dir

# The package __init__ re-exports the seed *function*, which shadows the
# submodule attribute — resolve the module through importlib instead.
seed_module = importlib.import_module("cognee.modules.seeding.seed")
seed = seed_module.seed


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (workspace / "MEMORY.md").write_text("# Memory\n")
    (workspace / "README.md").write_text("# Demo\n")
    return workspace


def make_claude_home(tmp_path: Path, workspace: Path) -> Path:
    claude_home = tmp_path / "claude-home"
    project_dir = claude_code_project_dir(workspace, claude_home)
    project_dir.mkdir(parents=True)
    transcript = project_dir / "session-1.jsonl"
    transcript.write_text('{"role": "user"}\n')
    stamp = time.time() - 60
    os.utime(transcript, (stamp, stamp))
    return claude_home


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Stub the ingestion surface and record every call."""
    calls = {"add": [], "cognify": []}

    async def fake_add(data, dataset_name=None, user=None, node_set=None, **kwargs):
        calls["add"].append((list(data), dataset_name, tuple(node_set or ())))

    async def fake_cognify(datasets=None, user=None, **kwargs):
        calls["cognify"].append(tuple(datasets or ()))

    async def no_existing_dataset(dataset_name, user):
        return False

    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)
    monkeypatch.setattr(seed_module, "_dataset_exists", no_existing_dataset)
    monkeypatch.setattr(seed_module, "_staging_dir", lambda: tmp_path / "staging")

    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace)
    monkeypatch.setattr(
        "cognee.modules.seeding.discovery.claude_code_project_dir",
        lambda ws, home=None: claude_code_project_dir(ws, claude_home),
    )
    return calls, workspace, tmp_path


@pytest.mark.asyncio
async def test_seed_stages_in_fast_first_value_order(harness):
    calls, workspace, _ = harness

    result = await seed(workspace, dataset_name="workspace")

    stage_names = [stage.name for stage in result.stages]
    assert stage_names == ["agent_memory", "workspace_docs", "session_logs", "codebase"]
    assert all(stage.added for stage in result.stages)
    assert all(stage.cognified for stage in result.stages)

    # Every add is tagged with its stage's node_set and lands in one dataset.
    assert [call[2] for call in calls["add"]] == [
        ("agent_memory",),
        ("workspace_docs",),
        ("session_logs",),
        ("codebase",),
    ]
    assert {call[1] for call in calls["add"]} == {"workspace"}
    assert calls["cognify"] == [("workspace",)] * 4

    # The codebase stage passes the workspace directory itself (the repo
    # resolver turns it into a manifest + documents downstream).
    assert calls["add"][-1][0] == [str(workspace)]
    assert result.ingested_anything


@pytest.mark.asyncio
async def test_seed_stages_out_of_root_sources(harness, monkeypatch):
    calls, workspace, tmp_path = harness
    # Only the workspace is an allowed root: the transcript under the fake
    # agent home must be staged into cognee's own storage before being added.
    monkeypatch.setenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", str(workspace))

    result = await seed(workspace)

    session_stage = next(stage for stage in result.stages if stage.name == "session_logs")
    staged_path = Path(session_stage.files[0])
    assert staged_path.is_relative_to(tmp_path / "staging")
    assert staged_path.name == "session-1.jsonl"
    assert staged_path.read_text() == '{"role": "user"}\n'


@pytest.mark.asyncio
async def test_cognify_failure_defers_but_keeps_ingesting(harness, monkeypatch):
    calls, workspace, _ = harness

    async def broken_cognify(datasets=None, user=None, **kwargs):
        calls["cognify"].append(tuple(datasets or ()))
        raise RuntimeError("no LLM key")

    monkeypatch.setattr("cognee.api.v1.cognify.cognify", broken_cognify)

    result = await seed(workspace)

    assert all(stage.added for stage in result.stages)
    assert not any(stage.cognified for stage in result.stages)
    # One failed attempt, then cognify is not retried per stage.
    assert len(calls["cognify"]) == 1
    assert "cognify deferred" in (result.stages[0].error or "")
    # Everything was still added — the data is there for a later cognify.
    assert len(calls["add"]) == len(result.stages)


@pytest.mark.asyncio
async def test_dry_run_discovers_without_ingesting(harness):
    calls, workspace, _ = harness

    result = await seed(workspace, dry_run=True)

    assert result.dry_run
    assert not result.stages
    assert not calls["add"] and not calls["cognify"]
    assert not result.plan.is_empty
    assert "Seed plan" in result.summary()


@pytest.mark.asyncio
async def test_existing_dataset_skips_unless_forced(harness, monkeypatch):
    calls, workspace, _ = harness

    async def dataset_exists(dataset_name, user):
        return True

    monkeypatch.setattr(seed_module, "_dataset_exists", dataset_exists)

    result = await seed(workspace)
    assert result.skipped_existing
    assert not calls["add"]

    forced = await seed(workspace, force=True)
    assert not forced.skipped_existing
    assert calls["add"]


@pytest.mark.asyncio
async def test_empty_plan_short_circuits(harness, tmp_path):
    calls, _, _ = harness
    barren = tmp_path / "barren"
    barren.mkdir()

    result = await seed(barren, include_session_logs=False)

    assert result.plan.is_empty
    assert not result.stages
    assert not calls["add"]
