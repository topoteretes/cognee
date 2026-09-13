import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.remember.remember import _maybe_presort_report, remember
from cognee.tasks.presort.models import PresortReport

PRESORT_MODULE = "cognee.modules.presort"

# Patch module OBJECTS, not dotted names. These packages re-export a function
# named like its module (``cognee.modules.presort.run_presort``,
# ``cognee.api.v1.add.add``, ``cognee.api.v1.remember.remember``); on Python
# 3.10 ``mock.patch`` resolves a dotted target attribute by attribute and lands
# on the function instead of the module.
add_module = importlib.import_module("cognee.api.v1.add.add")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
run_presort_module = importlib.import_module("cognee.modules.presort.run_presort")


@pytest.fixture(autouse=True)
def presort_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PRESORT_FOLDERS_ENABLED", raising=False)


@pytest.fixture
def sample_report():
    return PresortReport(scan_id="scan-1", root_path="/tmp/folder")


@pytest.mark.asyncio
async def test_dry_run_presort_routes_to_run_presort(sample_report):
    with patch(
        f"{PRESORT_MODULE}.run_presort", new=AsyncMock(return_value=sample_report)
    ) as run_mock:
        result = await remember(
            "/some/folder",
            dry_run="presort",
            use_llm=True,
            check_existing=False,
            dataset_prefix="dl_",
        )

    assert result is sample_report
    run_mock.assert_awaited_once()
    call_kwargs = run_mock.await_args.kwargs
    assert call_kwargs["use_llm"] is True
    assert call_kwargs["check_existing"] is False
    assert call_kwargs["dataset_prefix"] == "dl_"


@pytest.mark.asyncio
async def test_auto_apply_runs_apply_immediately(sample_report):
    apply_results = {"docs": "result"}
    with (
        patch(
            f"{PRESORT_MODULE}.run_presort", new=AsyncMock(return_value=sample_report)
        ) as run_mock,
        patch(
            f"{PRESORT_MODULE}.apply_presort", new=AsyncMock(return_value=apply_results)
        ) as apply_mock,
    ):
        result = await remember(
            "/some/folder", dry_run="presort", auto_apply=True, exclude_pii=True
        )

    run_mock.assert_awaited_once()
    apply_mock.assert_awaited_once()
    assert apply_mock.await_args.kwargs["exclude_pii"] is True
    assert result is sample_report
    assert result.apply_results == apply_results
    # apply_results is a live attachment — it never enters the serialized report
    assert "apply_results" not in result.to_dict()


@pytest.mark.asyncio
async def test_without_auto_apply_no_apply_call(sample_report):
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock(return_value=sample_report)),
        patch(f"{PRESORT_MODULE}.apply_presort", new=AsyncMock()) as apply_mock,
    ):
        result = await remember("/some/folder", dry_run="presort")

    apply_mock.assert_not_awaited()
    assert result.apply_results is None


@pytest.mark.asyncio
async def test_report_as_data_routes_to_apply(sample_report):
    with patch(
        f"{PRESORT_MODULE}.apply_presort", new=AsyncMock(return_value={"docs": "result"})
    ) as apply_mock:
        result = await remember(sample_report, apply_groups=["docs"], exclude_pii=True)

    assert result == {"docs": "result"}
    apply_mock.assert_awaited_once()
    call_kwargs = apply_mock.await_args.kwargs
    assert call_kwargs["groups"] == ["docs"]
    assert call_kwargs["exclude_pii"] is True


@pytest.mark.asyncio
async def test_report_dict_and_saved_path_detected(sample_report, tmp_path):
    assert _maybe_presort_report(sample_report.to_dict()) == sample_report

    saved = sample_report.save(tmp_path / "scan-1.presort.json")
    loaded = _maybe_presort_report(saved)
    assert loaded is not None
    assert loaded.scan_id == "scan-1"


def test_ordinary_inputs_not_detected_as_reports(tmp_path):
    plain_json = tmp_path / "data.json"
    plain_json.write_text('{"presort_report": true}')  # wrong suffix -> not a report
    assert _maybe_presort_report(str(plain_json)) is None
    assert _maybe_presort_report("just some text to remember") is None
    assert _maybe_presort_report({"some": "dict"}) is None
    assert _maybe_presort_report(["/a/path"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["true", "1", "yes"])
async def test_plain_folder_auto_presorts_when_enabled(tmp_path, sample_report, monkeypatch, flag):
    monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", flag)
    (tmp_path / "a.txt").write_text("hello")
    with (
        patch(
            f"{PRESORT_MODULE}.run_presort", new=AsyncMock(return_value=sample_report)
        ) as run_mock,
        patch(f"{PRESORT_MODULE}.apply_presort", new=AsyncMock(return_value={})) as apply_mock,
    ):
        result = await remember(str(tmp_path))

    run_mock.assert_awaited_once()
    apply_mock.assert_awaited_once()  # auto_apply implied for folder inputs
    assert result is sample_report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "remember_kwargs",
    [
        {"dataset_name": "my_project"},
        {"session_id": "s1"},
    ],
)
async def test_explicit_destination_skips_auto_presort(tmp_path, remember_kwargs, monkeypatch):
    monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", "true")
    (tmp_path / "a.txt").write_text("hello")
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch.object(
            remember_module, "_remember_inner", new=AsyncMock(return_value="inner")
        ) as inner_mock,
    ):
        result = await remember(str(tmp_path), **remember_kwargs)

    run_mock.assert_not_awaited()
    inner_mock.assert_awaited_once()
    assert result == "inner"


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, "false", "0", "no", "", "invalid"])
@pytest.mark.parametrize("remember_kwargs", [{}, {"dataset_name": "main_dataset"}])
async def test_folder_presort_requires_opt_in(tmp_path, monkeypatch, flag, remember_kwargs):
    (tmp_path / "a.txt").write_text("hello")
    if flag is not None:
        monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", flag)
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch.object(
            remember_module, "_remember_inner", new=AsyncMock(return_value="inner")
        ) as inner_mock,
    ):
        result = await remember(str(tmp_path), **remember_kwargs)

    run_mock.assert_not_awaited()
    assert result == "inner"
    assert inner_mock.await_args.args[1] == "main_dataset"


@pytest.mark.asyncio
async def test_code_project_folder_keeps_repo_route(tmp_path, monkeypatch):
    monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", "true")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
    (tmp_path / "main.py").write_text("print('x')")
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch.object(remember_module, "_remember_inner", new=AsyncMock(return_value="inner")),
    ):
        await remember(str(tmp_path))

    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_text_not_auto_presorted():
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch.object(remember_module, "_remember_inner", new=AsyncMock(return_value="inner")),
    ):
        await remember("Einstein was born in Ulm.")

    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_presort_downgrades_use_llm_without_key(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    from cognee.modules.presort.run_presort import run_presort

    with (
        patch.object(run_presort_module, "llm_is_configured", return_value=False),
        patch.object(run_presort_module, "_report_destination", return_value=None),
    ):
        report = await run_presort(str(tmp_path), use_llm=True, check_existing=False)

    assert report.used_llm is False
    assert any("deterministic pass only" in warning for warning in report.warnings)


@pytest.mark.asyncio
async def test_apply_without_llm_stages_with_add(sample_report):
    from cognee.modules.presort.apply_presort import apply_presort
    from cognee.tasks.presort.models import FileRecord, ProposedGroup

    sample_report.files = [FileRecord(path="/d/a.txt", name="a.txt")]
    sample_report.groups = [
        ProposedGroup(name="docs", dataset_name="docs", file_paths=["/d/a.txt"])
    ]

    with (
        patch("cognee.modules.presort.llm_availability.llm_is_configured", return_value=False),
        patch.object(add_module, "add", new=AsyncMock(return_value="add-result")) as add_mock,
        patch.object(remember_module, "remember", new=AsyncMock()) as remember_mock,
        patch(
            "cognee.tasks.presort.graph_apply.apply_presort_graph", new=AsyncMock()
        ) as graph_mock,
    ):
        results = await apply_presort(sample_report, apply_graph=True)

    remember_mock.assert_not_awaited()  # no cognify without an LLM
    graph_mock.assert_not_awaited()  # apply_graph skipped without embeddings
    add_mock.assert_awaited_once()
    assert add_mock.await_args.kwargs["dataset_name"] == "docs"
    assert results == {"docs": "add-result"}


@pytest.mark.asyncio
async def test_dry_run_presort_rejects_session_id():
    with pytest.raises(ValueError, match="session"):
        await remember("/some/folder", dry_run="presort", session_id="s1")


@pytest.mark.asyncio
async def test_apply_rejects_session_id(sample_report):
    with pytest.raises(ValueError, match="session_id"):
        await remember(sample_report, session_id="s1")


@pytest.mark.asyncio
async def test_dry_run_true_still_returns_estimate():
    estimate = object()
    with patch(
        "cognee.modules.cognify.estimator.estimate_remember_dry_run",
        new=AsyncMock(return_value=estimate),
    ) as estimate_mock:
        result = await remember("some text", dry_run=True)

    assert result is estimate
    estimate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_presort_end_to_end_deterministic(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("hello world")  # duplicate
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cv_ada.txt").write_text("resume, contact ada@example.com")

    with patch.object(run_presort_module, "_report_destination", return_value=None):
        report = await remember(str(tmp_path), dry_run="presort", check_existing=False)

    assert isinstance(report, PresortReport)
    assert {record.name for record in report.files} == {"a.txt", "b.txt", "cv_ada.txt"}
    assert len(report.duplicates) == 1
    assert any(finding.category == "email_address" for finding in report.pii)
    assert any(finding.category == "resume" for finding in report.pii)
    assert {group.name for group in report.groups} == {"docs", "documents"}
    assert report.summary()["junk"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_route", [False, True])
@pytest.mark.parametrize("filename", [None, ".DS_Store", ".hidden"])
async def test_empty_folder_apply_returns_report(tmp_path, monkeypatch, auto_route, filename):
    if filename:
        (tmp_path / filename).write_text("junk")
    if auto_route:
        monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", "true")
    with patch.object(run_presort_module, "_report_destination", return_value=None):
        report = await remember(
            str(tmp_path),
            dry_run=False if auto_route else "presort",
            auto_apply=True,
            check_existing=False,
        )

    assert isinstance(report, PresortReport)
    assert report.groups == []
    assert report.apply_results == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["report", "explicit", "automatic"])
@pytest.mark.parametrize("llm_available", [True, False])
async def test_presort_apply_preserves_ingestion_options(
    tmp_path, sample_report, monkeypatch, mode, llm_available
):
    from cognee.tasks.presort.models import FileRecord, ProposedGroup

    path = str(tmp_path / "a.txt")
    sample_report.files = [FileRecord(path=path, name="a.txt")]
    sample_report.groups = [ProposedGroup(name="docs", dataset_name="docs", file_paths=[path])]
    data = sample_report if mode == "report" else str(tmp_path)
    kwargs = {"dry_run": "presort", "auto_apply": True} if mode == "explicit" else {}
    if mode == "automatic":
        monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", "true")
    options = {
        "node_set": ["custom"],
        "node_set_extra": ["extra"],
        "vector_db_config": {"vector_db_provider": "pgvector"},
        "graph_db_config": {"graph_database_provider": "neo4j"},
        "preferred_loaders": ["TextLoader"],
        "data_per_batch": 3,
        "data_cache": False,
        "chunk_size": 100,
        "custom_prompt": "Extract people",
        "chunks_per_batch": 4,
        "self_improvement": False,
        "raise_on_error": False,
    }
    # Mock the actual ingestion boundary so this covers both remember's
    # presort routing and apply_presort's call into the standard pipeline.
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock(return_value=sample_report)),
        patch(
            "cognee.modules.presort.llm_availability.llm_is_configured", return_value=llm_available
        ),
        patch.object(
            remember_module, "_remember_inner", new=AsyncMock(return_value="result")
        ) as inner,
        patch.object(add_module, "add", new=AsyncMock(return_value="result")) as add,
    ):
        result = await remember(data, **kwargs, **options)

    if mode == "report":
        assert result == {"docs": "result"}
    else:
        assert result.apply_results == {"docs": "result"}
    called, unused = (inner, add) if llm_available else (add, inner)
    called.assert_awaited_once()
    unused.assert_not_awaited()
    actual = called.await_args.kwargs
    dataset = called.await_args.args[1] if llm_available else actual["dataset_name"]
    assert dataset == "docs"
    assert actual["node_set"] == ["presort", "docs", "custom", "extra"]
    assert actual["incremental_loading"] is True
    for key in (
        "vector_db_config",
        "graph_db_config",
        "preferred_loaders",
        "data_per_batch",
        "data_cache",
    ):
        assert actual[key] == options[key]
    if llm_available:
        for key in (
            "chunk_size",
            "custom_prompt",
            "chunks_per_batch",
            "self_improvement",
            "raise_on_error",
        ):
            assert actual[key] == options[key]
    else:
        assert actual["skip_connection_test"] is True
        assert "chunk_size" not in actual


@pytest.mark.asyncio
@pytest.mark.parametrize("apply_graph", [False, True])
async def test_invalid_explicit_group_still_raises(sample_report, apply_graph):
    with pytest.raises(ValueError, match="No groups to apply"):
        await remember(sample_report, apply_groups=["missing"], apply_graph=apply_graph)


@pytest.mark.asyncio
async def test_graph_only_apply_preserves_database_options(sample_report):
    with (
        patch("cognee.modules.presort.llm_availability.llm_is_configured", return_value=True),
        patch(
            "cognee.tasks.presort.graph_apply.apply_presort_graph",
            new=AsyncMock(return_value="graph"),
        ) as graph,
        patch.object(remember_module, "_remember_inner", new=AsyncMock()) as inner,
    ):
        result = await remember(
            sample_report,
            apply_groups=[],
            apply_graph=True,
            vector_db_config={"vector_db_provider": "pgvector"},
            graph_db_config={"graph_database_provider": "neo4j"},
        )
    assert result == {"presort_graph": "graph"}
    inner.assert_not_awaited()
    assert graph.await_args.kwargs["vector_db_config"] == {"vector_db_provider": "pgvector"}
    assert graph.await_args.kwargs["graph_db_config"] == {"graph_database_provider": "neo4j"}


def test_report_path_must_be_allowed_before_file_probe(tmp_path, monkeypatch):
    from pathlib import Path

    from cognee.infrastructure.files.utils import local_path_safety

    monkeypatch.setattr(
        local_path_safety, "get_allowed_local_file_roots", lambda: (tmp_path / "allowed",)
    )
    with (
        patch.object(Path, "is_file") as is_file,
        pytest.raises(ValueError, match="outside allowed roots"),
    ):
        _maybe_presort_report(tmp_path / "outside.presort.json")
    is_file.assert_not_called()


def test_auto_presort_checks_allowlist_before_probing_folder(tmp_path, monkeypatch):
    from pathlib import Path

    from cognee.infrastructure.files.utils import local_path_safety

    monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", "true")
    monkeypatch.setattr(
        local_path_safety, "get_allowed_local_file_roots", lambda: (tmp_path / "allowed",)
    )
    with patch.object(Path, "is_dir") as is_dir:
        assert not remember_module._should_auto_presort(
            tmp_path / "outside", "main_dataset", None, None, {}
        )
    is_dir.assert_not_called()
