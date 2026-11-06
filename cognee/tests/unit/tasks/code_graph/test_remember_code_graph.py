"""remember(<repo>, codegraph_config={"include_documents": False}) reports its repositories.

The code graph itself is add() + cognify()'s (SDK-793); what remember() owns is
the result mapping and the improve decision: one item per repository with its
kind, source, id, status and error, derived from the cognify run's per-item
results and the stored rows; the ``cognee.remember.code_graph`` telemetry
event; and no automatic improve when every item is a code repository. add(),
cognify() and the row lookup are stubbed, so no database, git or enola runs.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.api.v1.remember.remember import remember

remember_module = importlib.import_module("cognee.api.v1.remember.remember")

DATASET_ID = uuid4()
REPO_DIR = "/srv/repos/api"
CODE_ONLY = {"include_documents": False}


def _completed_entry(data_id):
    return {
        "run_info": SimpleNamespace(status="PipelineRunCompleted", error_message=None),
        "data_id": data_id,
    }


def _errored_entry(data_id, message):
    return {
        "run_info": SimpleNamespace(status="PipelineRunErrored", error_message=message),
        "data_id": data_id,
    }


def _already_completed_entry(data_id):
    return {
        "run_info": SimpleNamespace(status="PipelineRunAlreadyCompleted", error_message=None),
        "data_id": data_id,
    }


def _run(entries, status="PipelineRunCompleted", error_message=None):
    """A blocking cognify() return: {dataset_id: run info} with per-item results."""
    return {
        DATASET_ID: SimpleNamespace(
            status=status,
            pipeline_run_id=uuid4(),
            dataset_name="my_code",
            error_message=error_message,
            data_ingestion_info=entries,
            payload=None,
        )
    }


@pytest.fixture
def env(monkeypatch):
    """Stub everything around remember(): the rows are described by ``env.rows``."""
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("IMPROVE_AUTO_ENABLED", "true")
    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "validate_provider_config", Mock()
    )
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=DATASET_ID, name="my_code")
    # The rows add() stored, by id: system_metadata as the Data row carries it.
    rows: dict = {}
    monkeypatch.setattr(
        remember_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(user, [dataset])),
    )
    # add() reports the ids it stored; the fixture hands back every row it minted.
    add = AsyncMock(side_effect=lambda **_kwargs: SimpleNamespace(data_ids=list(rows)))
    cognify = AsyncMock(return_value=_run([]))
    monkeypatch.setattr("cognee.api.v1.add.add", add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", cognify)
    improve = AsyncMock(return_value=SimpleNamespace(status="completed", stages=[], error=None))
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.improve"), "improve", improve)
    telemetry = Mock()
    monkeypatch.setattr("cognee.shared.utils.send_telemetry", telemetry)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def execute(self, _statement):
            found = [
                SimpleNamespace(id=row_id, system_metadata=meta) for row_id, meta in rows.items()
            ]
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: found))

    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine",
        lambda: SimpleNamespace(get_async_session=_Session),
    )

    def repo_row(path=REPO_DIR, url=None):
        row_id = uuid4()
        rows[row_id] = {
            "source": "code_repo",
            "repo_path": path,
            **({"repo_url": url} if url else {}),
        }
        return row_id

    def document_row():
        row_id = uuid4()
        rows[row_id] = None
        return row_id

    return SimpleNamespace(
        user=user,
        add=add,
        cognify=cognify,
        improve=improve,
        telemetry=telemetry,
        repo_row=repo_row,
        document_row=document_row,
    )


@pytest.mark.asyncio
async def test_a_remembered_repository_is_one_item_with_its_identity(env):
    repo_id = env.repo_row(url="https://github.com/org/repo")
    env.cognify.return_value = _run([_completed_entry(repo_id)])

    result = await remember(REPO_DIR, dataset_name="my_code", codegraph_config=CODE_ONLY)

    assert result.status == "completed"
    assert result.pipeline_run_id
    assert result.items_processed == 1
    assert result.items == [
        {
            "id": str(repo_id),
            "pipeline_run_id": result.pipeline_run_id,
            "kind": "code_repository",
            "source": "https://github.com/org/repo",
            "path": REPO_DIR,
        }
    ]
    # The config reaches both halves of the build.
    assert env.add.await_args.kwargs["codegraph_config"] == CODE_ONLY
    assert env.cognify.await_args.kwargs["codegraph_config"] == CODE_ONLY


@pytest.mark.asyncio
async def test_an_all_repository_remember_skips_improve_and_sends_the_code_graph_event(env):
    repo_ids = [env.repo_row(path=f"/srv/repos/{name}") for name in ("a", "b")]
    env.cognify.return_value = _run([_completed_entry(repo_id) for repo_id in repo_ids])

    result = await remember(
        ["/srv/repos/a", "/srv/repos/b"], dataset_name="my_code", codegraph_config=CODE_ONLY
    )

    assert [item["kind"] for item in result.items] == ["code_repository"] * 2
    env.improve.assert_not_awaited()
    assert result.improve is None
    events = [call.args[0] for call in env.telemetry.call_args_list]
    assert "cognee.remember.code_graph" in events
    properties = next(
        call.kwargs["additional_properties"]
        for call in env.telemetry.call_args_list
        if call.args[0] == "cognee.remember.code_graph"
    )
    assert properties["repository_count"] == 2
    assert properties["index_vectors"] is False
    assert properties["dataset_name"] == "my_code"


@pytest.mark.asyncio
async def test_a_mixed_remember_still_runs_improve(env):
    repo_id, doc_id = env.repo_row(), env.document_row()
    env.cognify.return_value = _run([_completed_entry(repo_id), _completed_entry(doc_id)])

    result = await remember([REPO_DIR, "/srv/docs/notes.md"], dataset_name="my_code")

    assert [item.get("kind") for item in result.items] == ["code_repository", None]
    env.improve.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_unchanged_repository_is_still_reported_and_skips_improve(env):
    # Incremental loading skipped the row: the run reports it as already done.
    repo_id = env.repo_row()
    env.cognify.return_value = _run([_already_completed_entry(repo_id)])

    result = await remember(REPO_DIR, dataset_name="my_code", codegraph_config=CODE_ONLY)

    assert result.status == "completed"
    assert result.items[0]["id"] == str(repo_id)
    assert result.items_processed == 1
    env.improve.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_result_describes_this_call_not_the_datasets_other_rows(env):
    """The cognify run covers the dataset; items are cut to the rows add() stored."""
    mine, other = env.repo_row(path="/srv/repos/mine"), uuid4()
    env.add.side_effect = lambda **_kwargs: SimpleNamespace(data_ids=[mine])
    env.cognify.return_value = _run([_completed_entry(mine), _already_completed_entry(other)])

    result = await remember("/srv/repos/mine", dataset_name="my_code", codegraph_config=CODE_ONLY)

    assert [item["id"] for item in result.items] == [str(mine)]
    assert result.items_processed == 1


@pytest.mark.asyncio
async def test_a_no_op_re_remember_sends_no_code_graph_event(env):
    repo_id = env.repo_row()
    env.cognify.return_value = _run([_already_completed_entry(repo_id)])

    await remember(REPO_DIR, dataset_name="my_code", codegraph_config=CODE_ONLY)

    assert "cognee.remember.code_graph" not in [c.args[0] for c in env.telemetry.call_args_list]


@pytest.mark.asyncio
async def test_run_in_background_returns_running_then_completes(env):
    import asyncio

    repo_id = env.repo_row()
    release = asyncio.Event()

    async def _cognify(**_kwargs):
        await release.wait()
        return _run([_completed_entry(repo_id)])

    env.cognify.side_effect = _cognify

    result = await remember(
        REPO_DIR, dataset_name="my_code", codegraph_config=CODE_ONLY, run_in_background=True
    )

    assert result.status == "running"
    assert result.dataset_id == str(DATASET_ID)
    release.set()
    await result
    assert result.status == "completed"
    assert result.items[0] == {
        "id": str(repo_id),
        "pipeline_run_id": result.pipeline_run_id,
        "kind": "code_repository",
        "source": REPO_DIR,
        "path": REPO_DIR,
    }
    # The background run reports an errored build instead of raising.
    assert env.cognify.await_args.kwargs["raise_on_error"] is False


@pytest.mark.asyncio
async def test_a_blocking_remember_raises_a_failed_build_by_default(env):
    env.cognify.side_effect = RuntimeError("enola crashed")

    with pytest.raises(RuntimeError, match="enola crashed"):
        await remember(REPO_DIR, dataset_name="my_code", codegraph_config=CODE_ONLY)

    env.improve.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_clone_raises_out_of_add_before_anything_is_stored(env):
    env.add.side_effect = RuntimeError("clone failed")

    with pytest.raises(RuntimeError, match="clone failed"):
        await remember(
            "https://github.com/org/bad",
            dataset_name="my_code",
            codegraph_config={"treat_as_repository": True},
            raise_on_error=False,
        )

    env.cognify.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_id_is_rejected_with_code_options(env):
    with pytest.raises(ValueError, match="only supported for normal ingestion"):
        await remember(REPO_DIR, codegraph_config=CODE_ONLY, session_id="s1")


@pytest.mark.asyncio
async def test_a_bogus_credential_never_reaches_the_result_or_the_logs(env, caplog):
    import logging

    repo_id = env.repo_row(url="https://github.com/org/private")
    env.cognify.return_value = _run([_completed_entry(repo_id)])

    with caplog.at_level(logging.DEBUG):
        result = await remember(
            "https://github.com/org/private",
            dataset_name="my_code",
            codegraph_config={"repo_credentials": "ghs_bogus_token", **CODE_ONLY},
        )

    assert env.add.await_args.kwargs["codegraph_config"]["repo_credentials"] == "ghs_bogus_token"
    assert "ghs_bogus_token" not in str(result.to_dict())
    assert "ghs_bogus_token" not in repr(result)
    assert "ghs_bogus_token" not in caplog.text
