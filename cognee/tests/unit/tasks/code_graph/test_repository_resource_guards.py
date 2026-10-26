"""The disk budget, the enola run and the file selection each keep their own guarantees."""

import asyncio
import importlib
import json
import os
import sys
from contextlib import asynccontextmanager

import pytest

from cognee.tasks.code_graph import repo_resources as resources
from cognee.tasks.code_graph.repo_resources import (
    DocumentCopyBudget,
    RepositoryDiskConfig,
    RepositoryResourceError,
    check_repository_disk,
    disk_usage_bytes,
    repository_disk_budget,
)
from cognee.tasks.code_graph.repo_selection import (
    read_repository_snapshot,
    select_tree,
    selection_fingerprint,
    write_sparse_patterns,
)
from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

enola_module = importlib.import_module("cognee.tasks.code_graph.enola")
resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


@pytest.fixture(autouse=True)
def _isolated_storage(monkeypatch, tmp_path):
    storage = importlib.import_module("cognee.infrastructure.files.storage")
    monkeypatch.setattr(
        storage, "get_storage_config", lambda: {"data_root_directory": str(tmp_path / "documents")}
    )
    monkeypatch.delenv("COGNEE_REPOS_MAX_BYTES", raising=False)
    monkeypatch.delenv("COGNEE_REPOS_MIN_FREE_BYTES", raising=False)
    monkeypatch.setattr(resolve_module, "default_clones_dir", lambda: tmp_path)


# ---- the budget


def test_a_budget_equal_to_the_usage_is_exceeded(tmp_path):
    (tmp_path / "file").write_bytes(b"x" * 20000)
    used = disk_usage_bytes(tmp_path)

    with pytest.raises(RepositoryResourceError):
        check_repository_disk([tmp_path], RepositoryDiskConfig(max_bytes=used))
    assert check_repository_disk([tmp_path], RepositoryDiskConfig(max_bytes=used + 1)) == used


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_symlink_is_counted_as_a_link_not_as_its_target(tmp_path):
    big = tmp_path / "big"
    big.mkdir()
    (big / "file").write_bytes(b"x" * 200000)
    measured = tmp_path / "measured"
    measured.mkdir()
    (measured / "link").symlink_to(big, target_is_directory=True)

    assert disk_usage_bytes(measured) < 100000


def test_document_copies_scan_the_storage_once_and_add_up(tmp_path, monkeypatch):
    scans = []
    real = resources.disk_usage_bytes

    def counted(root):
        scans.append(root)
        return real(root)

    monkeypatch.setattr(resources, "disk_usage_bytes", counted)
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "1000000")

    budget = DocumentCopyBudget(tmp_path / "clones")
    scans_at_start = len(scans)
    for _ in range(300):
        budget.admit(1)

    assert len(scans) == scans_at_start
    with pytest.raises(RepositoryResourceError, match="document copy exceeds"):
        budget.admit(1000000)
    # The admitted copies count: the budget is not reset between documents.
    near = DocumentCopyBudget(tmp_path / "clones")
    near.admit(999000)
    with pytest.raises(RepositoryResourceError, match="document copy exceeds"):
        near.admit(999000)


@pytest.mark.asyncio
async def test_growth_during_the_run_stops_it_and_leaves_the_caller_uncancelled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "60000")
    clones = tmp_path / "clones"
    clones.mkdir()

    async def grow():
        for index in range(100):
            (clones / f"part{index}").write_bytes(b"x" * 20000)
            await asyncio.sleep(0.05)

    with pytest.raises(RepositoryResourceError, match="budget exceeded"):
        async with repository_disk_budget(clones):
            await grow()

    # Stopped while it was still growing, not noticed only when it finished.
    assert len(list(clones.iterdir())) < 30
    if sys.version_info >= (3, 11):
        assert asyncio.current_task().cancelling() == 0
    await asyncio.sleep(0)  # an outer await still works


@pytest.mark.asyncio
async def test_a_nested_budget_does_not_scan_again(tmp_path, monkeypatch):
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "1000000000")
    scans = []
    monkeypatch.setattr(resources, "check_repository_disk", lambda roots, config: scans.append(1))

    async with repository_disk_budget(tmp_path / "clones"):
        before = len(scans)
        async with repository_disk_budget(tmp_path / "clones"):
            pass
        assert len(scans) == before


# ---- the enola run


def _enola_repo(tmp_path, clone):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".enola").mkdir()
    (repo / ".enola" / "facts.jsonl").write_text("")
    if clone:
        (repo / ".git" / "cognee-snapshot.json").write_text("{}")
    return repo


@pytest.fixture
def enola(monkeypatch):
    seen = {"budgets": [], "env": None, "result": (0, b"", b"")}

    @asynccontextmanager
    async def budget(root):
        seen["budgets"].append(root)
        yield

    async def run_process(command, cwd=None, env=None, timeout=None):
        seen["env"] = env
        outcome = seen["result"]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(enola_module, "find_enola_binary", lambda: "enola")
    monkeypatch.setattr(enola_module, "repository_disk_budget", budget)
    monkeypatch.setattr(enola_module, "run_process", run_process)
    return seen


@pytest.mark.asyncio
async def test_a_local_checkout_is_neither_measured_nor_used_for_scratch(tmp_path, enola):
    repo = _enola_repo(tmp_path, clone=False)

    await enola_module.run_enola_generate(repo)

    assert enola["budgets"] == []
    assert not str(enola["env"]["TMPDIR"]).startswith(str(repo))
    assert enola["env"]["LC_ALL"] == "C"
    assert not list((repo / ".git").glob("cognee-enola-*"))


@pytest.mark.asyncio
async def test_a_managed_clone_is_measured_and_keeps_its_scratch_inside(tmp_path, enola):
    repo = _enola_repo(tmp_path, clone=True)

    await enola_module.run_enola_generate(repo)

    assert enola["budgets"] == [repo.parent]
    assert str(enola["env"]["TMPDIR"]).startswith(str(repo / ".git"))
    assert not list((repo / ".git").glob("cognee-enola-*"))


@pytest.mark.asyncio
async def test_enola_running_out_of_space_is_a_resource_error(tmp_path, enola):
    repo = _enola_repo(tmp_path, clone=False)
    enola["result"] = (1, b"", b"error: No space left on device (os error 28)")

    with pytest.raises(RepositoryResourceError):
        await enola_module.run_enola_generate(repo)

    assert not (repo / ".enola").exists()


@pytest.mark.asyncio
async def test_a_timed_out_enola_leaves_no_snapshot(tmp_path, enola):
    repo = _enola_repo(tmp_path, clone=False)
    enola["result"] = asyncio.TimeoutError()

    with pytest.raises(enola_module.EnolaSnapshotError, match="timed out"):
        await enola_module.run_enola_generate(repo, timeout=1)

    assert not (repo / ".enola").exists()


# ---- the clone


def test_git_never_downloads_lfs_content():
    assert resolve_module._git_env({})["GIT_LFS_SKIP_SMUDGE"] == "1"


@pytest.mark.asyncio
async def test_only_a_remote_resolve_runs_under_the_budget(tmp_path, monkeypatch):
    entered = []

    @asynccontextmanager
    async def budget(root):
        entered.append(root)
        yield

    async def resolved(*arguments):
        return "resolved"

    monkeypatch.setattr(resources, "repository_disk_budget", budget)
    monkeypatch.setattr(resolve_module, "_resolve_repo_source", resolved)

    assert await resolve_module.resolve_repo_source(str(tmp_path)) == "resolved"
    assert entered == []
    await resolve_module.resolve_repo_source("https://github.com/org/repo", clones_dir=tmp_path)
    assert entered == [tmp_path]


# ---- the selection


def _tree(*paths):
    return "".join(f"100644 blob {'a' * 40}\t{path}\0" for path in paths)


def test_generated_snapshots_and_media_are_never_selected():
    selected, skipped = select_tree(
        _tree(".enola/facts.jsonl", "docs/clip.mp4", "src/main.py"), include_documents=True
    )

    assert selected == ["src/main.py"]
    assert skipped["ignored_path"] == 1
    assert skipped["binary"] == 1


def test_sparse_patterns_are_anchored_and_an_empty_selection_selects_nothing(tmp_path):
    (tmp_path / ".git").mkdir()
    pattern_file = tmp_path / ".git" / "info" / "sparse-checkout"

    write_sparse_patterns(tmp_path, ["main.py", "dir/a b.py"])
    assert pattern_file.read_text().splitlines() == ["/main.py", "/dir/a\\ b.py"]

    write_sparse_patterns(tmp_path, [])
    assert pattern_file.read_text().splitlines() == ["!/*"]


def test_the_fingerprint_follows_the_paths_and_the_mode_but_not_their_order():
    base = selection_fingerprint(["a.py", "b.py"], True)

    assert selection_fingerprint(["b.py", "a.py"], True) == base
    assert selection_fingerprint(["a.py"], True) != base
    assert selection_fingerprint(["a.py", "b.py"], False) != base


def test_a_snapshot_record_that_is_not_an_object_is_refused(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "cognee-snapshot.json").write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(CodeRepositoryError, match="unreadable"):
        read_repository_snapshot(tmp_path)


def test_a_copy_that_reaches_the_budget_exactly_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "1000000")
    budget = DocumentCopyBudget(tmp_path / "clones")
    room = 1000000 - budget._used

    budget.admit(room - 1)
    fresh = DocumentCopyBudget(tmp_path / "clones")
    with pytest.raises(RepositoryResourceError, match="document copy exceeds"):
        fresh.admit(room)


def test_a_small_file_is_charged_the_blocks_it_occupies():
    class Info:
        st_size = 100
        st_blocks = 8

    assert resources.stat_disk_bytes(Info()) == 4096
    Info.st_blocks = 0
    assert resources.stat_disk_bytes(Info()) == 100


def test_git_output_is_requested_in_english():
    assert resolve_module._git_env({})["LC_ALL"] == "C"


@pytest.mark.asyncio
async def test_scratch_left_by_a_killed_run_is_removed_by_the_next_one(tmp_path, enola):
    repo = _enola_repo(tmp_path, clone=True)
    leftover = repo / ".git" / "cognee-enola-dead"
    leftover.mkdir()
    (leftover / "partial").write_text("x")

    await enola_module.run_enola_generate(repo)

    assert not leftover.exists()
