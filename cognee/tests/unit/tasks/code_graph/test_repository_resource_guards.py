"""The disk budget, the enola run and the file selection each keep their own guarantees."""

import asyncio
import importlib
import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

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


# ---- review follow-ups


@pytest.mark.parametrize("value", ["", "   "])
@pytest.mark.parametrize("name", ["COGNEE_REPOS_MAX_BYTES", "COGNEE_REPOS_MIN_FREE_BYTES"])
def test_an_empty_variable_means_no_limit(monkeypatch, name, value):
    """.env.template shows both variables empty: uncommented as shown they must not fail."""
    monkeypatch.setenv(name, value)

    config = resources.repository_disk_config()

    assert (config.max_bytes, config.min_free_bytes) == (None, None)


def test_the_507_message_names_the_setting_to_change():
    budget = RepositoryResourceError("budget exceeded", 100, used_bytes=150)
    free = RepositoryResourceError("insufficient free disk space", 100, free_bytes=40)
    physical = RepositoryResourceError("filesystem space or quota exhausted")

    assert "COGNEE_REPOS_MAX_BYTES" in budget.message
    assert "COGNEE_REPOS_MIN_FREE_BYTES" in free.message
    assert "COGNEE_REPOS_" not in physical.message
    assert {error.status_code for error in (budget, free, physical)} == {507}


@pytest.mark.parametrize(
    "stderr, full",
    [
        (b"fatal: No space left on device", True),
        (b"mkdir /r/.enola: no space left on device", True),  # Go prints it in lowercase
        (b"write /r/x: Disk quota exceeded", True),
        (b"write /r/x: disk quota exceeded", True),
        (b"fatal: repository not found", False),
    ],
)
def test_disk_full_is_recognised_in_any_letter_case(stderr, full):
    assert resources.reports_disk_full(stderr) is full


@pytest.mark.asyncio
async def test_a_lowercase_disk_full_of_enola_is_a_resource_error(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(enola_module, "find_enola_binary", lambda: "/bin/enola")
    monkeypatch.setattr(
        enola_module,
        "run_process",
        AsyncMock(return_value=(1, b"", b"mkdir /r/.enola: no space left on device\n")),
    )

    with pytest.raises(RepositoryResourceError) as raised:
        await enola_module.run_enola_generate(repo)

    assert raised.value.status_code == 507


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a POSIX non-root user")
def test_an_unreadable_directory_does_not_fail_the_scan(tmp_path):
    (tmp_path / "file").write_bytes(b"x" * 8192)
    locked = tmp_path / "lost+found"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert disk_usage_bytes(tmp_path) >= 8192
    finally:
        locked.chmod(0o700)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Task.uncancel needs Python 3.11")
@pytest.mark.asyncio
async def test_an_outer_cancellation_that_meets_the_guard_is_not_swallowed(tmp_path, monkeypatch):
    """The guard cancels the owner to stop it. If the owner is also cancelled from outside
    at that moment, the cancellation must still arrive."""
    monkeypatch.setenv("COGNEE_REPOS_MIN_FREE_BYTES", "1")
    owner = None

    def fires_and_gets_cancelled(roots, config):
        owner.cancel()  # the outer cancellation, arriving as the guard trips
        raise RepositoryResourceError("insufficient free disk space", 1, free_bytes=0)

    monkeypatch.setattr(resources, "_check_repository_free_space", fires_and_gets_cancelled)
    # Only the guard's own checks are replaced: the entry check must pass.
    real_check = resources.check_repository_disk
    entered = asyncio.Event()

    async def run():
        async with repository_disk_budget(tmp_path):
            entered.set()
            await asyncio.sleep(10)

    monkeypatch.setattr(resources, "check_repository_disk", lambda roots, config: None)
    owner = asyncio.create_task(run())
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert owner.cancelled()
    monkeypatch.setattr(resources, "check_repository_disk", real_check)


# ---- selection


def _tree(*paths):
    entries = []
    for path in paths:
        entries.append(f"100644 blob {'a' * 40}\t{path}")
    return "\0".join(entries) + "\0"


def test_directories_named_like_artefacts_are_skipped_at_any_depth_and_in_any_case():
    selected, skipped = select_tree(
        _tree(
            "src/app.py",
            "src/build/tasks.py",
            "pkg/DIST/a.py",
            ".ENOLA/facts.jsonl",
            "Build/b.py",
            "node_modules/x/y.js",
        ),
        include_documents=False,
    )

    assert selected == ["src/app.py"]
    assert skipped["ignored_path"] == 5


def test_a_file_that_merely_has_the_name_of_an_artefact_directory_is_kept():
    selected, skipped = select_tree(
        _tree("scripts/build", "tools/target", "venv", "coverage"), include_documents=False
    )

    assert selected == ["scripts/build", "tools/target", "venv", "coverage"]
    assert skipped["ignored_path"] == 0


def test_a_newline_in_a_selected_path_is_a_bad_request(tmp_path):
    with pytest.raises(CodeRepositoryError) as raised:
        write_sparse_patterns(tmp_path, ["a\nb.py"])

    assert raised.value.status_code == 400


def test_free_space_equal_to_the_reserve_is_insufficient(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024))
    config = RepositoryDiskConfig(min_free_bytes=1024)

    with pytest.raises(RepositoryResourceError):
        resources._check_free_space(tmp_path, config)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=1025))
    resources._check_free_space(tmp_path, config)
