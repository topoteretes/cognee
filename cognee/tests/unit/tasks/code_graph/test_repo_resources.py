"""The clone-root disk budget: configuration, admission, and the monitor that stops a run."""

import asyncio
import importlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.tasks.code_graph.repo_resources import (
    RepositoryDiskConfig,
    RepositoryResourceError,
    repository_disk_budget,
    repository_disk_config,
)
from cognee.tasks.code_graph.repo_selection import (
    SNAPSHOT_FILE,
    read_repository_snapshot,
    remove_lfs_pointers,
    remove_unselected_files,
    select_tree,
    write_repository_snapshot,
)
from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError


async def with_repository_disk_budget(operation, root):
    async with repository_disk_budget(root):
        return await operation()


@pytest.fixture(autouse=True)
def _isolated_budget(monkeypatch, tmp_path):
    storage = importlib.import_module("cognee.infrastructure.files.storage")
    monkeypatch.setattr(
        storage, "get_storage_config", lambda: {"data_root_directory": str(tmp_path / "documents")}
    )
    monkeypatch.delenv("COGNEE_REPOS_MAX_BYTES", raising=False)
    monkeypatch.delenv("COGNEE_REPOS_MIN_FREE_BYTES", raising=False)
    resolver = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
    monkeypatch.setattr(resolver, "default_clones_dir", lambda: tmp_path)


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1GB"])
def test_invalid_budget_is_not_silently_disabled(monkeypatch, value):
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", value)
    with pytest.raises(CogneeConfigurationError, match="COGNEE_REPOS_MAX_BYTES"):
        repository_disk_config()


@pytest.mark.asyncio
async def test_existing_other_clone_counts_toward_admission(tmp_path, monkeypatch):
    (tmp_path / "another-repo").mkdir()
    (tmp_path / "another-repo" / "data").write_bytes(b"x" * 16384)
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "8192")
    operation = AsyncMock()
    with pytest.raises(RepositoryResourceError, match="repository storage budget"):
        await with_repository_disk_budget(operation, tmp_path)
    operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_free_space_reserve_stops_before_start(tmp_path, monkeypatch):
    resources = importlib.import_module("cognee.tasks.code_graph.repo_resources")
    from types import SimpleNamespace

    monkeypatch.setenv("COGNEE_REPOS_MIN_FREE_BYTES", "1024")
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=512))
    operation = AsyncMock()
    with pytest.raises(RepositoryResourceError, match="insufficient free disk"):
        await with_repository_disk_budget(operation, tmp_path)
    operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchdog_waits_for_cancelled_operations_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "8192")
    cleaned = asyncio.Event()

    async def writer():
        output = tmp_path / "pack.tmp"
        try:
            output.write_bytes(b"x" * 16384)
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            output.unlink()
            cleaned.set()

    with pytest.raises(RepositoryResourceError):
        await asyncio.wait_for(with_repository_disk_budget(writer, tmp_path), timeout=5)
    assert cleaned.is_set()
    assert not (tmp_path / "pack.tmp").exists()


@pytest.mark.parametrize("tree", ["100644 blob " + "a" * 40 + "\tfoo.py", "bad\0", ""])
def test_incomplete_or_invalid_tree_is_not_silently_accepted(tree):
    if not tree:
        assert select_tree(tree, False)[0] == []  # An empty commit is valid.
    else:
        with pytest.raises(CodeRepositoryError, match="tree listing"):
            select_tree(tree, False)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process group termination")
async def test_enola_output_uses_same_disk_budget_and_reaps_process(tmp_path, monkeypatch):
    enola = importlib.import_module("cognee.tasks.code_graph.enola")
    root = tmp_path / "clones"
    repo = root / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "cognee-snapshot.json").write_text("{}")
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "65536")
    monkeypatch.setattr(enola, "find_enola_binary", lambda: sys.executable)
    real_spawn = asyncio.create_subprocess_exec
    processes = []

    async def spawn(*args, **kwargs):
        script = (
            "import time; from pathlib import Path; "
            "Path('.enola').mkdir(); "
            "Path('.enola/facts.jsonl').write_bytes(b'x' * 131072); time.sleep(60)"
        )
        process = await real_spawn(sys.executable, "-c", script, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(enola.asyncio, "create_subprocess_exec", spawn)
    try:
        with pytest.raises(RepositoryResourceError):
            await asyncio.wait_for(enola.run_enola_generate(repo), timeout=5)
        assert processes and processes[0].returncode is not None
        assert not (repo / ".enola").exists()
        assert not list((repo / ".git").glob("cognee-enola-*"))
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.communicate()


@pytest.mark.asyncio
async def test_no_implicit_repository_size_limit(tmp_path):
    assert RepositoryDiskConfig().max_bytes is None
    assert RepositoryDiskConfig().min_free_bytes is None
    operation = AsyncMock(return_value="completed")
    assert await with_repository_disk_budget(operation, tmp_path) == "completed"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process group termination")
async def test_filter_refusal_interrupts_transfer_before_process_finishes(tmp_path, monkeypatch):
    resolver = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
    real_spawn = asyncio.create_subprocess_exec
    processes = []

    async def spawn(*args, **kwargs):
        script = (
            "import sys,time; from pathlib import Path; "
            "sys.stderr.write('warning: filtering not recognized '); sys.stderr.flush(); "
            "time.sleep(0.05); "
            "sys.stderr.write('by server, ignoring\\n'); sys.stderr.flush(); "
            "time.sleep(2); Path('full-pack').write_bytes(b'x' * 131072)"
        )
        process = await real_spawn(sys.executable, "-c", script, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(resolver.asyncio, "create_subprocess_exec", spawn)
    try:
        with pytest.raises(CodeRepositoryError, match="partial clone filter"):
            await asyncio.wait_for(resolver._exec_git(["clone"], cwd=tmp_path), timeout=5)
        assert processes and processes[0].returncode is not None
        assert not (tmp_path / "full-pack").exists()
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.communicate()


def test_budget_errors_name_what_they_measured():
    used = RepositoryResourceError("clone storage budget exceeded", 100, used_bytes=150)
    free = RepositoryResourceError("insufficient free disk space", 100, free_bytes=40)

    assert (used.used_bytes, used.free_bytes) == (150, None)
    assert (free.used_bytes, free.free_bytes) == (None, 40)
    assert "150 bytes used; budget 100 bytes" in used.message
    assert "40 bytes free; at least 100 bytes required" in free.message


def test_a_corrupt_snapshot_record_is_a_repository_error(tmp_path):
    (tmp_path / ".git").mkdir()
    assert read_repository_snapshot(tmp_path) is None
    write_repository_snapshot(tmp_path, {"commit": "abc"})
    assert read_repository_snapshot(tmp_path) == {"commit": "abc"}
    assert not list((tmp_path / ".git").glob("*.tmp"))

    (tmp_path / ".git" / SNAPSHOT_FILE).write_text('{"commit": ')
    with pytest.raises(CodeRepositoryError, match="unreadable"):
        read_repository_snapshot(tmp_path)


def test_a_selected_path_missing_on_disk_is_a_repository_error(tmp_path):
    (tmp_path / "present.py").write_text("x = 1\n")

    assert remove_lfs_pointers(tmp_path, ["present.py"]) == 0
    with pytest.raises(CodeRepositoryError, match="was not written"):
        remove_lfs_pointers(tmp_path, ["present.py", "missing/file.py"])


def test_lfs_pointers_are_removed_and_counted_and_real_files_are_kept(tmp_path):
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 9\n"
    (tmp_path / "data.csv").write_text(pointer)
    (tmp_path / "keep.py").write_text("x = 1\n")

    assert remove_lfs_pointers(tmp_path, ["data.csv", "keep.py"]) == 1

    assert not (tmp_path / "data.csv").exists()
    assert (tmp_path / "keep.py").read_text() == "x = 1\n"


@pytest.mark.skipif(os.name == "nt", reason="Creating symlinks requires Windows privileges")
@pytest.mark.parametrize("link", ["directory", "snapshot", "pending"])
def test_snapshot_metadata_cannot_escape_the_checkout(tmp_path, link):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "repo-outside"
    outside.mkdir()
    original = '{"commit": "private"}'
    secret = outside / SNAPSHOT_FILE
    secret.write_text(original)
    git_dir = repo / ".git"
    if link == "directory":
        git_dir.symlink_to(outside, target_is_directory=True)
    else:
        git_dir.mkdir()
        name = SNAPSHOT_FILE + (".tmp" if link == "pending" else "")
        (git_dir / name).symlink_to(secret)

    if link != "pending":
        with pytest.raises(CodeRepositoryError, match="outside its checkout"):
            read_repository_snapshot(repo)
    with pytest.raises(CodeRepositoryError, match="outside its checkout"):
        write_repository_snapshot(repo, {"commit": "replacement"})
    assert secret.read_text() == original
    assert list(outside.iterdir()) == [secret]


@pytest.mark.asyncio
async def test_the_monitor_pauses_in_proportion_to_a_scan(monkeypatch, tmp_path):
    """A slow scan of a big clone root must not turn the guard into a busy loop."""
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "1000000000000")
    resources = importlib.import_module("cognee.tasks.code_graph.repo_resources")
    scans = []

    def slow_scan(root, config):
        scans.append(1)
        time.sleep(0.05)

    monkeypatch.setattr(resources, "_check_repository_size", slow_scan)

    async def operation():
        await asyncio.sleep(0.4)
        return "done"

    assert await with_repository_disk_budget(operation, tmp_path) == "done"
    # Two scans bracket the run, the monitor's first one runs at once, and a 0.05s scan
    # then pauses 0.5s: a fixed 0.1s pause would have run about four scans in 0.4s.
    assert len(scans) <= 3


def test_document_storage_counts_toward_shared_budget(tmp_path, monkeypatch):
    from cognee.tasks.code_graph.repo_resources import DocumentCopyBudget

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "old.txt").write_bytes(b"x" * 16384)
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "8192")
    with pytest.raises(RepositoryResourceError, match="repository storage budget"):
        DocumentCopyBudget(tmp_path / "clones")
    assert (documents / "old.txt").exists()


def test_document_copy_checks_known_size_before_writing(tmp_path, monkeypatch):
    from cognee.tasks.code_graph.repo_resources import DocumentCopyBudget

    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "8192")
    with pytest.raises(RepositoryResourceError, match="document copy exceeds"):
        DocumentCopyBudget(tmp_path / "clones").admit(16384)
    assert not (tmp_path / "documents").exists()


def test_nested_storage_roots_are_counted_once(tmp_path, monkeypatch):
    from cognee.tasks.code_graph.repo_resources import repository_storage_roots

    assert repository_storage_roots(tmp_path) == [tmp_path.resolve()]


@pytest.mark.asyncio
async def test_reader_rejects_other_selection_at_same_commit(tmp_path, monkeypatch):
    from cognee.tasks.code_graph import code_repo

    monkeypatch.setattr(code_repo, "repository_commit", AsyncMock(return_value="a" * 40))
    (tmp_path / ".git").mkdir()
    write_repository_snapshot(tmp_path, {"selection_fingerprint": "new-selection"})
    with pytest.raises(code_repo.RepositoryChangedError):
        async with code_repo.repository_read(
            tmp_path, remote=True, expected_commit="a" * 40, expected_selection="old-selection"
        ):
            pytest.fail("Mismatched checkout reached a reader")


@pytest.mark.asyncio
async def test_reader_rejects_invalidated_snapshot(tmp_path, monkeypatch):
    from cognee.tasks.code_graph import code_repo

    monkeypatch.setattr(code_repo, "repository_commit", AsyncMock(return_value="a" * 40))
    with pytest.raises(code_repo.RepositoryChangedError):
        async with code_repo.repository_read(
            tmp_path, remote=True, expected_commit="a" * 40, expected_selection="old-selection"
        ):
            pytest.fail("Invalidated checkout reached a reader")


def test_file_uri_storage_is_included_in_budget(tmp_path, monkeypatch):
    from cognee.infrastructure.files import storage
    from cognee.tasks.code_graph.repo_resources import (
        DocumentCopyBudget,
        repository_storage_roots,
    )

    documents = tmp_path / "uri-documents"
    documents.mkdir()
    (documents / "existing").write_bytes(b"x" * 16384)
    monkeypatch.setattr(
        storage, "get_storage_config", lambda: {"data_root_directory": documents.as_uri()}
    )
    assert documents in repository_storage_roots(tmp_path / "clones")
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "8192")
    with pytest.raises(RepositoryResourceError):
        DocumentCopyBudget(tmp_path / "clones")


@pytest.mark.asyncio
async def test_reader_requires_selection_for_pre_partial_manifest(tmp_path, monkeypatch):
    from cognee.tasks.code_graph import code_repo

    monkeypatch.setattr(code_repo, "repository_commit", AsyncMock(return_value="a" * 40))
    with pytest.raises(code_repo.RepositoryChangedError):
        async with code_repo.repository_read(tmp_path, remote=True, expected_commit="a" * 40):
            pytest.fail("Pre-partial manifest must be re-added")


@pytest.mark.asyncio
@pytest.mark.parametrize("number", [28, 122])
async def test_physical_exhaustion_without_configured_limit_is_resource_error(tmp_path, number):
    import errno

    number = errno.ENOSPC if number == 28 else errno.EDQUOT
    with pytest.raises(RepositoryResourceError) as raised:
        async with repository_disk_budget(tmp_path):
            raise OSError(number, "write failed")
    assert raised.value.status_code == 507
    assert raised.value.limit_bytes is None
    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_other_filesystem_errors_are_not_resource_errors(tmp_path):
    import errno

    with pytest.raises(PermissionError):
        async with repository_disk_budget(tmp_path):
            raise PermissionError(errno.EACCES, "denied")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stderr", [b"fatal: No space left on device", b"fatal: Disk quota exceeded"]
)
async def test_git_filesystem_exhaustion_is_resource_error(monkeypatch, stderr):
    from cognee.tasks.code_graph import resolve_repo

    monkeypatch.setattr(resolve_repo, "run_process", AsyncMock(return_value=(128, b"", stderr)))
    with pytest.raises(RepositoryResourceError) as raised:
        await resolve_repo._exec_git(["clone", "https://example.com/repo.git"])
    assert raised.value.status_code == 507
    assert raised.value.limit_bytes is None


@pytest.mark.asyncio
async def test_free_space_monitor_interrupts_while_size_scan_is_blocked(tmp_path, monkeypatch):
    """A size scan that is stuck must not delay the free-space check that stops the run."""
    resources = importlib.import_module("cognee.tasks.code_graph.repo_resources")
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "1000000")
    monkeypatch.setenv("COGNEE_REPOS_MIN_FREE_BYTES", "1024")
    scanning = threading.Event()
    release_scan = threading.Event()
    scan_finished = threading.Event()
    cleaned = asyncio.Event()
    free = 2048
    calls = 0

    def scan(roots, config):
        nonlocal calls
        calls += 1
        if calls > 1:
            scanning.set()
            try:
                assert release_scan.wait(5), "test did not release the size scan"
            finally:
                scan_finished.set()
        return 0

    monkeypatch.setattr(resources, "_check_repository_size", scan)
    # resources.shutil is the shutil module: this patch is process-wide while the test runs.
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))

    async def operation():
        nonlocal free
        assert await asyncio.to_thread(scanning.wait, 5)
        free = 512
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    try:
        with pytest.raises(RepositoryResourceError, match="insufficient free disk") as raised:
            await asyncio.wait_for(with_repository_disk_budget(operation, tmp_path), timeout=3)
        assert raised.value.free_bytes == 512
        assert raised.value.status_code == 507
        assert cleaned.is_set()
        # The scan only reads, so teardown does not wait for it (a cancelled writer is
        # waited for, see resolve_repo): it is still blocked here and drained below.
        assert not scan_finished.is_set()
    finally:
        release_scan.set()
        assert await asyncio.to_thread(scan_finished.wait, 5)


def test_free_space_is_checked_before_scanning_size(tmp_path, monkeypatch):
    """Running out of space is reported first, without paying for a size scan."""
    resources = importlib.import_module("cognee.tasks.code_graph.repo_resources")
    scan = Mock()
    monkeypatch.setattr(resources, "disk_usage_bytes", scan)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=512))
    with pytest.raises(RepositoryResourceError, match="insufficient free disk"):
        resources.check_repository_disk(
            [tmp_path], RepositoryDiskConfig(max_bytes=8192, min_free_bytes=1024)
        )
    scan.assert_not_called()


@pytest.mark.asyncio
async def test_both_monitors_failing_do_not_cancel_cleanup_twice(tmp_path, monkeypatch):
    """The second monitor failing while the owner cleans up must not cancel it again."""
    resources = importlib.import_module("cognee.tasks.code_graph.repo_resources")
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", "8192")
    monkeypatch.setenv("COGNEE_REPOS_MIN_FREE_BYTES", "1024")
    writing = threading.Event()
    cleaned = asyncio.Event()

    def free_space(roots, config):
        if writing.is_set():
            raise RepositoryResourceError("test limit reached", 1024, free_bytes=512)

    def size(roots, config):
        if writing.is_set():
            # Fails while the owner is cleaning up after the free-space failure.
            time.sleep(0.05)
            raise RepositoryResourceError("test limit reached", 1024, free_bytes=512)

    monkeypatch.setattr(resources, "_check_repository_size", size)
    monkeypatch.setattr(resources, "_check_repository_free_space", free_space)

    async def operation():
        writing.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.1)
            cleaned.set()

    with pytest.raises(RepositoryResourceError, match="test limit reached"):
        await asyncio.wait_for(with_repository_disk_budget(operation, tmp_path), timeout=5)
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_free_space_is_checked_even_when_every_executor_thread_is_busy(tmp_path, monkeypatch):
    """The free-space check runs on the event loop, so blocked threads cannot delay it."""
    resources = importlib.import_module("cognee.tasks.code_graph.repo_resources")
    monkeypatch.setenv("COGNEE_REPOS_MIN_FREE_BYTES", "1024")
    free = 2048
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    release = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)

    async def operation():
        nonlocal free
        asyncio.get_running_loop().set_default_executor(pool)
        pool.submit(release.wait, 5)  # the only worker is now busy
        free = 512
        await asyncio.Event().wait()

    try:
        with pytest.raises(RepositoryResourceError, match="insufficient free disk"):
            await asyncio.wait_for(with_repository_disk_budget(operation, tmp_path), timeout=3)
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_files_outside_the_selection_are_removed_from_the_work_tree(tmp_path):
    """A path that stops being a file (a file replaced by a submodule) keeps its old content
    in the work tree: the checkout must contain exactly the selection."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("kept")
    (tmp_path / "keep.py").write_text("x = 1\n")
    (tmp_path / "stale.py").write_text("old\n")
    (tmp_path / "gone").mkdir()
    (tmp_path / "gone" / "old.py").write_text("old\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "keep.py").write_text("y = 2\n")

    assert remove_unselected_files(tmp_path, ["keep.py", "sub/keep.py"]) == 2

    assert sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()
    ) == [
        ".git/config",
        "keep.py",
        "sub/keep.py",
    ]
    assert not (tmp_path / "gone").exists()
