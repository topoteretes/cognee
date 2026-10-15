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


DATASET_ID = uuid4()


def _run_info(status, **fields):
    return SimpleNamespace(status=status, dataset_name=None, payload=None, **fields)


@pytest.fixture
def failing_repo_add(monkeypatch):
    """Stub add()/cognify() per repository.

    add() fails (a clone error) for any spec containing 'bad', reports no stored
    ids for one containing 'noids', and otherwise stores one row whose id is
    derived from the spec; cognify() records its
    calls and fails the build (an errored run, as an enola failure is
    reported) of any row whose spec contains 'unbuildable'.
    """
    calls = {"add": [], "cognify": []}
    spec_by_id = {}

    async def fake_add(*args, **kwargs):
        spec = kwargs["data"]
        calls["add"].append(spec)
        if "bad" in str(spec):
            raise RuntimeError("clone failed")
        if "noids" in str(spec):
            return _run_info("PipelineRunCompleted", data_ingestion_info=[])
        data_id = uuid4()
        spec_by_id[data_id] = str(spec)
        return _run_info(
            "PipelineRunCompleted",
            data_ingestion_info=[
                {"run_info": _run_info("PipelineRunCompleted"), "data_id": data_id}
            ],
        )

    async def fake_cognify(*args, **kwargs):
        calls["cognify"].append(kwargs)
        data_ids = kwargs.get("data_ids") or []
        run_id = uuid4()
        if any("unbuildable" in spec_by_id.get(data_id, "") for data_id in data_ids):
            return {
                DATASET_ID: _run_info(
                    "PipelineRunErrored", pipeline_run_id=run_id, error_message="enola failed"
                )
            }
        return {
            DATASET_ID: _run_info(
                "PipelineRunCompleted",
                pipeline_run_id=run_id,
                data_ingestion_info=[
                    {"run_info": _run_info("PipelineRunCompleted"), "data_id": data_id}
                    for data_id in data_ids
                ],
            )
        }

    async def no_identity(result):
        return None

    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)
    monkeypatch.setattr(remember_module, "_attach_item_identity", no_identity)
    return calls


DECLARED = {"codegraph_config": {"treat_as_repository": True}}
BAD = "https://bitbucket.org/acme/bad"
BAD_ITEM = {"kind": "code_repo", "source": BAD, "status": "errored", "error": "clone failed"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data, cognify_runs",
    [
        # One repository behaves like many: raise_on_error=False reports it.
        (BAD, 0),
        (["https://bitbucket.org/acme/good", BAD], 1),
    ],
)
async def test_one_failing_repository_does_not_stop_the_others(
    failing_repo_add, data, cognify_runs
):
    result = await _remember(data, raise_on_error=False, **DECLARED)

    assert failing_repo_add["add"] == (data if isinstance(data, list) else [data])
    assert len(failing_repo_add["cognify"]) == cognify_runs
    assert result.status == "errored"
    assert [item for item in result.items if item.get("status") == "errored"] == [BAD_ITEM]
    assert f"{BAD}: clone failed" in result.error


@pytest.mark.asyncio
async def test_a_failed_build_rolls_back_only_its_own_repository(failing_repo_add):
    # Each repository builds in its own scoped cognify run: repository A's
    # failed build is an errored item naming A, and B's graph is kept.
    result = await _remember(
        ["https://bitbucket.org/acme/unbuildable-a", "https://bitbucket.org/acme/b"],
        raise_on_error=False,
        **DECLARED,
    )

    cognify_calls = failing_repo_add["cognify"]
    assert len(cognify_calls) == 2
    # Scoped to that repository's own rows, never the whole dataset.
    assert [len(call["data_ids"]) for call in cognify_calls] == [1, 1]
    assert cognify_calls[0]["data_ids"] != cognify_calls[1]["data_ids"]
    assert all(call["raise_on_error"] is False for call in cognify_calls)

    assert result.status == "errored"
    built = [item for item in result.items if item.get("status") != "errored"]
    errored = [item for item in result.items if item.get("status") == "errored"]
    assert [item["id"] for item in built] == [str(cognify_calls[1]["data_ids"][0])]
    assert built[0]["pipeline_run_id"] == result.pipeline_run_id
    assert len(errored) == 1
    assert errored[0]["kind"] == "code_repo"
    assert errored[0]["source"] == "https://bitbucket.org/acme/unbuildable-a"
    assert errored[0]["error"] == "enola failed"
    assert errored[0]["pipeline_run_id"] != result.pipeline_run_id
    assert result.items_processed == 1
    assert "https://bitbucket.org/acme/unbuildable-a: enola failed" == result.error


@pytest.mark.asyncio
async def test_every_repository_failing_skips_cognify(failing_repo_add):
    result = await _remember(
        ["https://bitbucket.org/acme/bad1", "https://bitbucket.org/acme/bad2"],
        raise_on_error=False,
        **DECLARED,
    )

    assert failing_repo_add["cognify"] == []
    assert result.status == "errored"
    assert [item["source"] for item in result.items] == [
        "https://bitbucket.org/acme/bad1",
        "https://bitbucket.org/acme/bad2",
    ]
    assert all(item["kind"] == "code_repo" for item in result.items)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data, cognify_runs",
    [(BAD, 0), ([BAD, "https://bitbucket.org/acme/good"], 1)],
)
async def test_background_run_isolates_repository_failures(failing_repo_add, data, cognify_runs):
    result = await _remember(data, run_in_background=True, **DECLARED)
    await result._task

    assert len(failing_repo_add["add"]) == (len(data) if isinstance(data, list) else 1)
    assert len(failing_repo_add["cognify"]) == cognify_runs
    assert result.status == "errored"
    assert [item for item in result.items if item.get("status") == "errored"] == [BAD_ITEM]


@pytest.mark.asyncio
async def test_an_add_reporting_no_ids_never_builds_the_whole_dataset(failing_repo_add):
    result = await _remember(
        ["https://bitbucket.org/acme/noids", "https://bitbucket.org/acme/good"],
        raise_on_error=False,
        **DECLARED,
    )

    assert [len(call["data_ids"]) for call in failing_repo_add["cognify"]] == [1]
    errored = [item for item in result.items if item.get("status") == "errored"]
    assert [item["source"] for item in errored] == ["https://bitbucket.org/acme/noids"]
    assert "reported no stored data" in errored[0]["error"]


@pytest.mark.asyncio
async def test_without_incremental_loading_and_data_cache_repositories_build_together(
    failing_repo_add, caplog
):
    """Such an add reports no ids, so nothing could scope a per-repository cognify:
    the batch is added and built in one run instead, with a warning."""
    data = ["https://bitbucket.org/acme/a", "https://bitbucket.org/acme/b"]
    with caplog.at_level("WARNING"):
        await _remember(
            data, raise_on_error=False, incremental_loading=False, data_cache=False, **DECLARED
        )

    assert failing_repo_add["add"] == [data]
    ((cognify_call,),) = [failing_repo_add["cognify"]]
    assert "data_ids" not in cognify_call
    assert "report no data ids" in caplog.text


@pytest.mark.asyncio
async def test_blocking_raise_on_error_still_raises_on_the_first_failure(failing_repo_add):
    # Isolation only where failures are reported.
    with pytest.raises(RuntimeError, match="clone failed"):
        await _remember([BAD, "https://bitbucket.org/acme/good"], **DECLARED)

    assert failing_repo_add["cognify"] == []


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
async def test_repo_credentials_never_reach_logs_or_the_result(failing_repo_add, caplog):
    # A failed clone of a credentialed spec is logged and reported: neither may
    # carry the token, whether it came in codegraph_config or the URL userinfo.
    import logging

    token = "ghs_secret_token_123"
    with caplog.at_level(logging.DEBUG):
        result = await _remember(
            [
                "https://bitbucket.org/acme/good",
                f"https://x-access-token:{token}@bitbucket.org/acme/bad",
            ],
            codegraph_config={"treat_as_repository": True, "repo_credentials": token},
            raise_on_error=False,
        )

    assert token not in caplog.text
    assert token not in repr(result.to_dict())
    assert token not in (result.error or "")


@pytest.mark.asyncio
async def test_mistyped_codegraph_values_are_rejected_before_any_work(permanent_pipeline):
    with pytest.raises(TypeError, match="must be a bool"):
        await _remember(
            "https://github.com/acme/api", codegraph_config={"include_documents": "false"}
        )

    assert permanent_pipeline == {}
