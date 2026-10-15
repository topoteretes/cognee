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


@pytest.mark.asyncio
async def test_codegraph_config_reaches_add_and_cognify(permanent_pipeline):
    config = {"index_vectors": True, "repo_credentials": "tok123"}

    result = await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        codegraph_config=config,
    )

    assert result.status == "completed"
    assert permanent_pipeline["add"]["codegraph_config"] == config
    assert permanent_pipeline["cognify"]["codegraph_config"] == config


@pytest.mark.asyncio
@pytest.mark.parametrize("index_vectors", [False, True])
async def test_top_level_spellings_fold_into_codegraph_config(permanent_pipeline, index_vectors):
    result = await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        index_vectors=index_vectors,
        repo_credentials="tok123",
    )

    assert result.status == "completed"
    expected = {"index_vectors": index_vectors, "repo_credentials": "tok123"}
    assert permanent_pipeline["add"]["codegraph_config"] == expected
    assert permanent_pipeline["cognify"]["codegraph_config"] == expected
    for call in ("add", "cognify"):
        assert "index_vectors" not in permanent_pipeline[call]
        assert "repo_credentials" not in permanent_pipeline[call]


@pytest.mark.asyncio
async def test_explicit_codegraph_config_wins_over_top_level_spelling(permanent_pipeline):
    await remember_module.remember(
        "/some/repo",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        index_vectors=False,
        codegraph_config={"index_vectors": True},
    )

    assert permanent_pipeline["cognify"]["codegraph_config"] == {"index_vectors": True}


@pytest.mark.asyncio
async def test_unknown_codegraph_config_key_is_rejected(permanent_pipeline):
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: index_vector"):
        await remember_module.remember(
            "/some/repo",
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            codegraph_config={"index_vector": True},
        )

    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_with_session_id(permanent_pipeline):
    # The session path never runs cognify, so the option would be ignored.
    with pytest.raises(ValueError, match="only supported for normal ingestion"):
        await remember_module.remember(
            "note",
            session_id="s1",
            user=SimpleNamespace(id=uuid4()),
            index_vectors=True,
        )

    assert permanent_pipeline == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"repo_credentials": "tok123"},
        {"codegraph_config": {"include_documents": True}},
    ],
)
async def test_removed_code_content_type_is_ignored(permanent_pipeline, kwargs, caplog):
    # Accepted so existing callers do not break, and ignored: the call is a
    # plain remember(data) -- no warning, no translation into codegraph_config.
    import logging
    import warnings

    dataset_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=dataset_id,
        user=user,
        self_improvement=False,
        **kwargs,
    )
    plain = dict(permanent_pipeline)
    permanent_pipeline.clear()

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        result = await remember_module.remember(
            "https://github.com/acme/api",
            dataset_id=dataset_id,
            user=user,
            self_improvement=False,
            content_type="code",
            **kwargs,
        )

    assert result.status == "completed"
    assert permanent_pipeline == plain
    assert "content_type" not in permanent_pipeline["add"]
    assert "content_type" not in permanent_pipeline["cognify"]
    assert [w for w in caught if "content_type" in str(w.message)] == []
    assert "content_type" not in caplog.text


@pytest.mark.asyncio
async def test_code_content_type_with_session_id_takes_the_session_path(
    permanent_pipeline, monkeypatch
):
    stored = []

    async def fake_add_to_session(session_id, data, user):
        stored.append((session_id, data))

    monkeypatch.setattr(remember_module, "_add_to_session", fake_add_to_session)

    result = await remember_module.remember(
        "/some/repo",
        session_id="s1",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        content_type="code",
    )

    assert result.status == "session_stored"
    assert stored == [("s1", "/some/repo")]
    assert permanent_pipeline == {}


@pytest.mark.asyncio
async def test_include_documents_reaches_add(permanent_pipeline):
    await remember_module.remember(
        "/some/repo",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        codegraph_config={"include_documents": False},
    )

    assert permanent_pipeline["add"]["codegraph_config"] == {"include_documents": False}


@pytest.mark.asyncio
async def test_other_unsupported_content_types_are_still_rejected(permanent_pipeline):
    with pytest.raises(ValueError, match="Unsupported remember content_type"):
        await remember_module.remember(
            "/some/repo",
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            content_type="pdf",
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
async def test_code_content_type_runs_improve_by_default(permanent_pipeline, improve_calls):
    # Ignored means ignored: the improve default is the plain call's.
    await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        content_type="code",
    )

    assert len(improve_calls) == 1


@pytest.mark.asyncio
async def test_plain_remember_still_runs_improve_by_default(permanent_pipeline, improve_calls):
    await remember_module.remember(
        "some text",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
    )

    assert len(improve_calls) == 1


@pytest.mark.asyncio
async def test_explicit_index_vectors_false_is_not_a_codegraph_option(permanent_pipeline):
    # index_vectors=False is the default restated: it must not trip the checks
    # that reject codegraph options with session_id or dry_run.
    from cognee.tasks.code_graph.config import requests_codegraph_options

    assert not requests_codegraph_options({"index_vectors": False})
    assert not requests_codegraph_options({"include_documents": True})
    assert requests_codegraph_options({"include_documents": False})
    assert remember_module._should_auto_presort is not None

    with pytest.raises(ValueError) as error:
        await remember_module.remember(
            "note",
            session_id="s1",
            user=SimpleNamespace(id=uuid4()),
            index_vectors=False,
            dry_run=True,
        )
    # Rejected for dry_run with a session, not for a codegraph option.
    assert "codegraph_config" not in str(error.value)


DATASET_ID = uuid4()


def _run_info(status, **fields):
    return SimpleNamespace(status=status, dataset_name=None, payload=None, **fields)


@pytest.fixture
def failing_repo_add(monkeypatch):
    """Stub add()/cognify() per repository.

    add() fails (a clone error) for any spec containing 'bad' and otherwise
    stores one row whose id is derived from the spec; cognify() records its
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


@pytest.mark.asyncio
async def test_one_failing_repository_does_not_stop_the_others(failing_repo_add):
    result = await remember_module.remember(
        ["https://bitbucket.org/acme/good", "https://bitbucket.org/acme/bad"],
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        raise_on_error=False,
        **DECLARED,
    )

    assert failing_repo_add["add"] == [
        "https://bitbucket.org/acme/good",
        "https://bitbucket.org/acme/bad",
    ]
    assert len(failing_repo_add["cognify"]) == 1
    assert result.status == "errored"
    errored = [item for item in result.items if item.get("status") == "errored"]
    assert errored == [
        {
            "kind": "code_repo",
            "source": "https://bitbucket.org/acme/bad",
            "status": "errored",
            "error": "clone failed",
        }
    ]
    assert "https://bitbucket.org/acme/bad: clone failed" in result.error


@pytest.mark.asyncio
async def test_a_failed_build_rolls_back_only_its_own_repository(failing_repo_add):
    # Each repository builds in its own scoped cognify run: repository A's
    # failed build is an errored item naming A, and B's graph is kept.
    result = await remember_module.remember(
        ["https://bitbucket.org/acme/unbuildable-a", "https://bitbucket.org/acme/b"],
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
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
async def test_a_single_failing_repository_is_reported_not_raised(failing_repo_add):
    # One repository behaves like many: raise_on_error=False reports it.
    result = await remember_module.remember(
        "https://bitbucket.org/acme/bad",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        raise_on_error=False,
        **DECLARED,
    )

    assert failing_repo_add["cognify"] == []
    assert result.status == "errored"
    assert result.items == [
        {
            "kind": "code_repo",
            "source": "https://bitbucket.org/acme/bad",
            "status": "errored",
            "error": "clone failed",
        }
    ]
    assert result.error == "https://bitbucket.org/acme/bad: clone failed"


@pytest.mark.asyncio
async def test_a_single_failing_repository_in_the_background_names_it(failing_repo_add):
    result = await remember_module.remember(
        "https://bitbucket.org/acme/bad",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        run_in_background=True,
        **DECLARED,
    )
    await result._task

    assert result.status == "errored"
    assert [item["source"] for item in result.items] == ["https://bitbucket.org/acme/bad"]
    assert result.items[0]["error"] == "clone failed"


@pytest.mark.asyncio
async def test_every_repository_failing_skips_cognify(failing_repo_add):
    result = await remember_module.remember(
        ["https://bitbucket.org/acme/bad1", "https://bitbucket.org/acme/bad2"],
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
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
async def test_background_run_isolates_repository_failures(failing_repo_add):
    result = await remember_module.remember(
        ["https://bitbucket.org/acme/bad", "https://bitbucket.org/acme/good"],
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        run_in_background=True,
        **DECLARED,
    )
    await result._task

    assert len(failing_repo_add["add"]) == 2
    assert len(failing_repo_add["cognify"]) == 1
    assert result.status == "errored"


@pytest.mark.asyncio
async def test_blocking_raise_on_error_still_raises_on_the_first_failure(failing_repo_add):
    # Isolation only where failures are reported.
    with pytest.raises(RuntimeError, match="clone failed"):
        await remember_module.remember(
            ["https://bitbucket.org/acme/bad", "https://bitbucket.org/acme/good"],
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            **DECLARED,
        )

    assert failing_repo_add["cognify"] == []


@pytest.mark.asyncio
async def test_new_style_call_keeps_code_repo_kind(permanent_pipeline, monkeypatch):
    async def fake_attach(result):
        result.items = [{"id": "a", "kind": "code_repo", "source": "https://github.com/acme/api"}]

    monkeypatch.setattr(remember_module, "_attach_item_identity", fake_attach)

    result = await remember_module.remember(
        "https://github.com/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
    )

    assert result.items[0]["kind"] == "code_repo"


@pytest.mark.asyncio
async def test_codegraph_config_is_rejected_for_memory_entries(permanent_pipeline):
    from cognee.memory import QAEntry

    with pytest.raises(ValueError, match="not supported for memory entries"):
        await remember_module.remember(
            QAEntry(question="q", answer="a"),
            session_id="s1",
            user=SimpleNamespace(id=uuid4()),
            codegraph_config={"treat_as_repository": True},
        )


@pytest.mark.asyncio
async def test_repo_credentials_never_reach_logs_or_the_result(failing_repo_add, caplog):
    # A failed clone of a credentialed spec is logged and reported: neither may
    # carry the token, whether it came in codegraph_config or the URL userinfo.
    import logging

    token = "ghs_secret_token_123"
    with caplog.at_level(logging.DEBUG):
        result = await remember_module.remember(
            [
                "https://bitbucket.org/acme/good",
                f"https://x-access-token:{token}@bitbucket.org/acme/bad",
            ],
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            codegraph_config={"treat_as_repository": True, "repo_credentials": token},
            raise_on_error=False,
        )

    assert token not in caplog.text
    assert token not in repr(result.to_dict())
    assert token not in (result.error or "")


@pytest.mark.asyncio
async def test_mistyped_codegraph_values_are_rejected_before_any_work(permanent_pipeline):
    with pytest.raises(TypeError, match="must be a bool"):
        await remember_module.remember(
            "https://github.com/acme/api",
            dataset_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
            codegraph_config={"include_documents": "false"},
        )

    assert permanent_pipeline == {}
