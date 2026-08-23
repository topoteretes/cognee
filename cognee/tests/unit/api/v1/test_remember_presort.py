from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.remember.remember import _maybe_presort_report, remember
from cognee.tasks.presort.models import PresortReport

PRESORT_MODULE = "cognee.modules.presort"


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
async def test_plain_folder_auto_presorts(tmp_path, sample_report):
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
async def test_explicit_destination_skips_auto_presort(tmp_path, remember_kwargs):
    (tmp_path / "a.txt").write_text("hello")
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch(
            "cognee.api.v1.remember.remember._remember_inner", new=AsyncMock(return_value="inner")
        ) as inner_mock,
    ):
        result = await remember(str(tmp_path), **remember_kwargs)

    run_mock.assert_not_awaited()
    inner_mock.assert_awaited_once()
    assert result == "inner"


@pytest.mark.asyncio
async def test_auto_presort_env_kill_switch(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello")
    monkeypatch.setenv("PRESORT_FOLDERS_ENABLED", "false")
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch(
            "cognee.api.v1.remember.remember._remember_inner", new=AsyncMock(return_value="inner")
        ),
    ):
        await remember(str(tmp_path))

    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_code_project_folder_keeps_repo_route(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
    (tmp_path / "main.py").write_text("print('x')")
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch(
            "cognee.api.v1.remember.remember._remember_inner", new=AsyncMock(return_value="inner")
        ),
    ):
        await remember(str(tmp_path))

    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_text_not_auto_presorted():
    with (
        patch(f"{PRESORT_MODULE}.run_presort", new=AsyncMock()) as run_mock,
        patch(
            "cognee.api.v1.remember.remember._remember_inner", new=AsyncMock(return_value="inner")
        ),
    ):
        await remember("Einstein was born in Ulm.")

    run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_presort_downgrades_use_llm_without_key(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    from cognee.modules.presort.run_presort import run_presort

    with (
        patch("cognee.modules.presort.run_presort.llm_is_configured", return_value=False),
        patch("cognee.modules.presort.run_presort._report_destination", return_value=None),
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
        patch("cognee.api.v1.add.add.add", new=AsyncMock(return_value="add-result")) as add_mock,
        patch("cognee.api.v1.remember.remember.remember", new=AsyncMock()) as remember_mock,
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

    with patch(f"{PRESORT_MODULE}.run_presort._report_destination", return_value=None):
        report = await remember(str(tmp_path), dry_run="presort", check_existing=False)

    assert isinstance(report, PresortReport)
    assert {record.name for record in report.files} == {"a.txt", "b.txt", "cv_ada.txt"}
    assert len(report.duplicates) == 1
    assert any(finding.category == "email_address" for finding in report.pii)
    assert any(finding.category == "resume" for finding in report.pii)
    assert {group.name for group in report.groups} == {"docs", "documents"}
    assert report.summary()["junk"] == 1
