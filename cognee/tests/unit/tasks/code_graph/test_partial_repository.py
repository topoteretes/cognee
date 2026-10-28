"""Partial clone selection against a real filtered Git upload-pack server."""

import importlib
import json
import os
import shutil
import subprocess
from unittest.mock import AsyncMock

import pytest

from cognee.tests.unit.tasks.code_graph.test_resolve_repo_real_git import Origin, _git

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
_URL = "https://github.com/org/partial"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


@pytest.fixture(autouse=True)
def isolated_git(tmp_path, monkeypatch):
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_TEMPLATE_DIR", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    for name in ("COGNEE_REPOS_MAX_BYTES", "COGNEE_REPOS_MIN_FREE_BYTES"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    remote = Origin(tmp_path / "remote")
    _git(remote.bare, "config", "uploadpack.allowFilter", "true")
    real_exec = resolve_module._exec_git

    async def exec_git(args, cwd=None, env=None):
        return await real_exec(
            [remote.bare.resolve().as_uri() if arg == _URL else arg for arg in args], cwd, env
        )

    monkeypatch.setattr(resolve_module, "_exec_git", exec_git)
    real_origin_url = resolve_module._origin_url

    async def origin_url(clone):
        stored = await real_origin_url(clone)
        return _URL if stored == remote.bare.resolve().as_uri() else stored

    monkeypatch.setattr(resolve_module, "_origin_url", origin_url)
    ssrf = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")
    monkeypatch.setattr(ssrf, "validate_outbound_url", AsyncMock())
    return remote


@pytest.fixture
def clones(tmp_path):
    return tmp_path / "clones"


def git_output(cwd, *args):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def local_objects(repo):
    # Enumerating the local object database cannot trigger a lazy fetch, unlike cat-file -e.
    return set(
        git_output(
            repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)"
        ).splitlines()
    )


@pytest.mark.asyncio
async def test_excluded_large_blob_is_not_downloaded(origin, clones):
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    (origin.work / "weights.bin").write_bytes(os.urandom(4 * 1024 * 1024))
    (origin.work / "main.py").write_text("print('hello')\n")
    origin.commit("main", "artifact")
    blob = git_output(origin.work, "rev-parse", "HEAD:weights.bin")

    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)

    assert (target / "main.py").read_text() == "print('hello')\n"
    assert not (target / "weights.bin").exists()
    assert blob not in local_objects(target)
    assert git_output(target, "rev-parse", "--is-shallow-repository") == "true"
    assert git_output(target, "config", "remote.origin.partialclonefilter") == "blob:none"
    snapshot = read_repository_snapshot(target)
    assert snapshot is not None
    assert snapshot["commit"] == git_output(origin.work, "rev-parse", "HEAD")
    assert snapshot["skipped"]["binary"] == 1


@pytest.mark.asyncio
async def test_mode_change_on_same_commit_changes_snapshot_and_checkout(origin, clones):
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    (origin.work / "diagram.png").write_bytes(b"diagram-document")
    origin.commit("main", "document")
    target = await resolve_module.resolve_repo_source(
        _URL, clones_dir=clones, include_documents=False
    )
    code_only = read_repository_snapshot(target)
    assert code_only is not None
    assert not (target / "diagram.png").exists()
    assert (
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones, include_documents=True)
        == target
    )
    with_documents = read_repository_snapshot(target)
    assert with_documents is not None
    assert (target / "diagram.png").read_bytes() == b"diagram-document"
    assert with_documents["commit"] == code_only["commit"]
    assert with_documents["selection_fingerprint"] != code_only["selection_fingerprint"]
    await resolve_module.resolve_repo_source(_URL, clones_dir=clones, include_documents=False)
    assert not (target / "diagram.png").exists()
    restored = read_repository_snapshot(target)
    assert restored is not None
    assert restored["selection_fingerprint"] == code_only["selection_fingerprint"]


@pytest.mark.asyncio
async def test_server_without_filter_fails_without_publishing_a_clone(origin, clones):
    _git(origin.bare, "config", "uploadpack.allowFilter", "false")
    with pytest.raises(resolve_module.CodeRepositoryError, match="filter|partial clone"):
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    assert not resolve_module.clone_dir(_URL, clones_dir=clones).exists()
    assert not list(clones.glob(".*.tmp-*"))


@pytest.mark.asyncio
async def test_sparse_checkout_preserves_literal_paths_and_extractor_inputs(origin, clones):
    names = ["space name.py", "[literal].py", "trailing .py", ".gitignore", "config.custom"]
    if os.name != "nt":
        names += ["star*.py", "question?.py", "back\\slash.py"]
    names += [".github/workflows/check.yml", "nested/source.py"]
    for name in names:
        path = origin.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# " + name)
    ignored = origin.work / "node_modules" / "dependency.py"
    ignored.parent.mkdir()
    ignored.write_text("# generated dependency")
    origin.commit("main", "paths")
    excluded_blob = git_output(origin.work, "rev-parse", "HEAD:node_modules/dependency.py")
    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    for name in names:
        assert (target / name).read_text() == "# " + name
    assert not (target / "node_modules").exists()
    assert excluded_blob not in local_objects(target)


@pytest.mark.asyncio
async def test_a_selected_lfs_pointer_is_skipped_and_never_runs_smudge(origin, clones, tmp_path):
    marker = tmp_path / "smudged"
    (origin.work / ".gitattributes").write_text("large.py filter=lfs\n")
    (origin.work / "large.py").write_text(
        "version https://git-lfs.github.com/spec/v1\n" + "oid sha256:" + "a" * 64 + "\nsize 123\n"
    )
    origin.commit("main", "lfs")
    # Global config also reaches clone's checkout; source repo config alone would not.
    _git(origin.work, "config", "--global", "filter.lfs.smudge", f"touch '{marker}'")
    _git(origin.work, "config", "--global", "filter.lfs.required", "true")
    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    assert not marker.exists()
    # The pointer is not content: it is removed from the checkout and counted, and the
    # rest of the repository is ingested.
    assert not (target / "large.py").exists()
    assert (target / "BRANCH.txt").exists()
    snapshot = json.loads((target / ".git" / "cognee-snapshot.json").read_text())
    assert snapshot["skipped"]["lfs_pointer"] == 1
    # The selected file count is of the files that remain on disk.
    on_disk = [
        path
        for path in target.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(target).parts
    ]
    assert snapshot["selected_files"] == len(on_disk)
    assert not list(clones.glob(".*.tmp-*"))


@pytest.mark.asyncio
async def test_full_clone_is_replaced_with_filtered_objects(origin, clones):
    (origin.work / "weights.bin").write_bytes(os.urandom(1024 * 1024))
    origin.commit("main", "full-cache")
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    clones.mkdir()
    _git(clones, "clone", "--depth", "1", origin.bare.resolve().as_uri(), str(target))
    blob = git_output(origin.work, "rev-parse", "HEAD:weights.bin")
    assert blob in local_objects(target)

    assert await resolve_module.resolve_repo_source(_URL, clones_dir=clones) == target

    assert blob not in local_objects(target)
    assert not (target / "weights.bin").exists()
    assert (target / "BRANCH.txt").read_text().strip() == "full-cache"


@pytest.mark.asyncio
async def test_failed_full_clone_replacement_preserves_original(origin, clones):
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    clones.mkdir()
    _git(clones, "clone", "--depth", "1", origin.bare.resolve().as_uri(), str(target))
    commit = git_output(target, "rev-parse", "HEAD")
    _git(origin.bare, "config", "uploadpack.allowFilter", "false")
    with pytest.raises(resolve_module.CodeRepositoryError, match="filter|partial clone"):
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    assert git_output(target, "rev-parse", "HEAD") == commit
    assert (target / "BRANCH.txt").read_text().strip() == "main-1"
    assert not list(clones.glob(".*.tmp-*"))


@pytest.mark.asyncio
async def test_symlinks_and_submodules_are_not_materialized(origin, clones):
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    commit = git_output(origin.work, "rev-parse", "HEAD")
    _git(origin.work, "update-index", "--add", "--cacheinfo", f"160000,{commit},dependency")
    link_blob = (
        subprocess.check_output(
            ["git", "hash-object", "-w", "--stdin"], cwd=origin.work, input=b"../../secret.py"
        )
        .decode()
        .strip()
    )
    _git(origin.work, "update-index", "--add", "--cacheinfo", f"120000,{link_blob},link.py")
    _git(origin.work, "commit", "-qm", "special entries")
    _git(origin.work, "push", "-q", "origin", "main")
    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    snapshot = read_repository_snapshot(target)
    assert snapshot is not None
    assert snapshot["skipped"]["submodule"] == 1
    assert snapshot["skipped"]["symlink"] == 1
    assert not (target / "link.py").is_symlink()
    assert not (target / "dependency").exists()
    assert link_blob not in local_objects(target)


@pytest.mark.parametrize(
    "tree",
    [
        "100644 blob " + "a" * 40 + "\tmain.py",
        "malformed\0",
        "100644 blob invalid\tmain.py\0",
        "100644 blob " + "a" * 40 + "\t../secret.py\0",
        "100644 blob " + "a" * 40 + "\t/secret.py\0",
        "100644 blob " + "a" * 40 + "\ta//b.py\0",
        ("100644 blob " + "a" * 40 + "\tmain.py\0") * 2,
        "040000 tree " + "a" * 40 + "\tdirectory\0",
    ],
)
def test_malformed_tree_is_rejected(tree):
    from cognee.tasks.code_graph.repo_selection import select_tree

    with pytest.raises(resolve_module.CodeRepositoryError):
        select_tree(tree, False)


def test_empty_tree_is_valid():
    from cognee.tasks.code_graph.repo_selection import select_tree

    selected, skipped = select_tree("", False)
    assert selected == []
    assert all(count == 0 for count in skipped.values())


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_disk_monitor_does_not_move_clone_into_another_task(origin, clones, monkeypatch):
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", str(1024 * 1024 * 1024))
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    # add() owns this lock before the resolver starts. The budget monitor must
    # retain that task ownership instead of waiting on its parent's lock.
    async with resolve_module._clone_lock(target, clones_dir=clones):
        assert await resolve_module.resolve_repo_source(_URL, clones_dir=clones) == target
    assert (target / "BRANCH.txt").read_text().strip() == "main-1"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="Windows filenames cannot contain newlines")
async def test_selected_newline_path_fails_before_checkout(origin, clones):
    (origin.work / "line\nbreak.py").write_text("# source")
    origin.commit("main", "newline")
    with pytest.raises(resolve_module.CodeRepositoryError, match="newline"):
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    assert not resolve_module.clone_dir(_URL, clones_dir=clones).exists()
    assert not list(clones.glob(".*.tmp-*"))


@pytest.mark.asyncio
async def test_global_url_rewrite_cannot_bypass_filtered_transport(tmp_path, clones, monkeypatch):
    remote = Origin(tmp_path / "rewrite-remote")
    _git(remote.bare, "config", "uploadpack.allowFilter", "true")
    (remote.work / "large.bin").write_bytes(os.urandom(1024 * 1024))
    remote.commit("main", "large artifact")
    blob = git_output(remote.work, "rev-parse", "HEAD:large.bin")
    _git(remote.work, "config", "--global", f"url.{remote.bare.resolve()}.insteadOf", _URL)
    ssrf = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")
    monkeypatch.setattr(ssrf, "validate_outbound_url", AsyncMock())

    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)

    assert blob not in local_objects(target)
    assert git_output(target, "rev-parse", "--is-shallow-repository") == "true"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
@pytest.mark.parametrize("refresh", [False, True])
async def test_cancelled_sparse_writer_finishes_before_clone_cleanup(
    origin, clones, monkeypatch, refresh
):
    import asyncio
    import threading

    selection = importlib.import_module("cognee.tasks.code_graph.repo_selection")
    read_repository_snapshot = selection.read_repository_snapshot
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    if refresh:
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = selection.write_sparse_patterns

    def paused_writer(*args):
        entered.set()
        try:
            assert release.wait(10), "Test did not release the sparse writer"
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(selection, "write_sparse_patterns", paused_writer)
    task = asyncio.create_task(resolve_module.resolve_repo_source(_URL, clones_dir=clones))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        # Repeated cancellation must not release ownership of a still-running write.
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "Resolver abandoned a live filesystem writer"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        if refresh:
            # A cancelled refresh keeps the clone: its snapshot record is gone, so the next
            # run replaces it instead of trusting a half-updated checkout.
            assert target.exists()
            assert read_repository_snapshot(target) is None
        else:
            assert not target.exists()
        assert not list(clones.glob(".*.tmp-*"))
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if entered.is_set():
            assert finished.is_set()
    monkeypatch.setattr(selection, "write_sparse_patterns", original)
    assert await resolve_module.resolve_repo_source(_URL, clones_dir=clones) == target
    assert (target / "BRANCH.txt").read_text().strip() == "main-1"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_cancelled_full_cache_retirement_publishes_valid_replacement(
    origin, clones, monkeypatch
):
    import asyncio
    import threading

    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    clones.mkdir()
    _git(clones, "clone", "--depth", "1", origin.bare.resolve().as_uri(), str(target))
    entered, release = threading.Event(), threading.Event()
    original = resolve_module._remove_clone

    def paused_removal(path):
        if ".old-" in path.name:  # the retired clone: the replacement is already published
            entered.set()
            assert release.wait(10), "Test did not release cache retirement"
        return original(path)

    monkeypatch.setattr(resolve_module, "_remove_clone", paused_removal)
    task = asyncio.create_task(resolve_module.resolve_repo_source(_URL, clones_dir=clones))
    reader_entered = asyncio.Event()

    async def read_after_publication():
        async with resolve_module._clone_lock(target, clones_dir=clones):
            reader_entered.set()
            assert read_repository_snapshot(target)["commit"] == git_output(
                target, "rev-parse", "HEAD"
            )
            assert (target / "BRANCH.txt").read_text().strip() == "main-1"

    reader = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        reader = asyncio.create_task(read_after_publication())
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert not reader_entered.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await reader
        assert not list(clones.glob(".*.tmp-*"))
        assert not list(clones.glob(".*.old-*"))
    finally:
        release.set()
        await asyncio.gather(
            task, *([reader] if reader is not None else []), return_exceptions=True
        )
    assert await resolve_module.resolve_repo_source(_URL, clones_dir=clones) == target


@pytest.mark.asyncio
async def test_a_failed_removal_of_the_retired_clone_never_fails_the_publish(
    origin, clones, monkeypatch
):
    """The old clone is renamed aside before it is deleted: a deletion that fails (or a
    process killed meanwhile) leaves a retired directory, never a half-deleted target."""
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    clones.mkdir()
    _git(clones, "clone", "--depth", "1", origin.bare.resolve().as_uri(), str(target))
    original = resolve_module._remove_clone

    def failing_removal(path):
        if ".old-" in path.name:
            raise PermissionError("busy")
        return original(path)

    monkeypatch.setattr(resolve_module, "_remove_clone", failing_removal)

    assert await resolve_module.resolve_repo_source(_URL, clones_dir=clones) == target

    assert (target / ".git").is_dir() and (target / "BRANCH.txt").exists()
    assert len(list(clones.glob(".*.old-*"))) == 1
    monkeypatch.setattr(resolve_module, "_remove_clone", original)
    assert await resolve_module.resolve_repo_source(_URL, clones_dir=clones) == target
    assert not list(clones.glob(".*.old-*"))


@pytest.mark.asyncio
async def test_the_snapshot_record_is_invalidated_before_the_checkout_is_touched(
    origin, clones, monkeypatch
):
    """A refresh that fails after it started to update the checkout must not leave the old
    record behind, or an earlier manifest would be accepted for a half-updated clone."""
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    assert read_repository_snapshot(target) is not None
    seen = []

    async def failing_checkout(repo, *args, **kwargs):
        seen.append(read_repository_snapshot(repo))
        raise resolve_module.CodeRepositoryError(message="checkout failed")

    monkeypatch.setattr(resolve_module, "_checkout_snapshot", failing_checkout)
    with pytest.raises(resolve_module.CodeRepositoryError, match="checkout failed"):
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones)

    assert seen == [None]
    assert read_repository_snapshot(target) is None


@pytest.mark.asyncio
async def test_a_budget_error_during_a_refresh_frees_the_clone(origin, clones, monkeypatch):
    from cognee.tasks.code_graph.repo_resources import RepositoryResourceError

    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)

    async def out_of_space(*args, **kwargs):
        raise RepositoryResourceError("repository storage budget exceeded", 1, used_bytes=2)

    monkeypatch.setattr(resolve_module, "_checkout_snapshot", out_of_space)
    with pytest.raises(RepositoryResourceError):
        await resolve_module.resolve_repo_source(_URL, clones_dir=clones)

    assert not target.exists()


@pytest.mark.asyncio
async def test_a_file_replaced_by_a_submodule_does_not_stay_in_the_work_tree(origin, clones):
    """git leaves the old file in place when its path becomes a gitlink: the checkout must
    contain exactly the selection, so Enola never reads a file the fingerprint omits."""
    (origin.work / "y.py").write_text("old = True\n")
    _git(origin.work, "add", "y.py")
    origin.commit("main", "with-file")
    target = await resolve_module.resolve_repo_source(_URL, clones_dir=clones)
    assert (target / "y.py").read_text() == "old = True\n"

    _git(origin.work, "rm", "-q", "-f", "y.py")
    _git(origin.work, "update-index", "--add", "--cacheinfo", f"160000,{'a' * 40},y.py")
    # Not origin.commit(): it stages the work tree again and would drop the gitlink.
    _git(origin.work, "commit", "-q", "-m", "with-gitlink")
    _git(origin.work, "push", "-q", "origin", "main")
    await resolve_module.resolve_repo_source(_URL, clones_dir=clones)

    assert not (target / "y.py").exists()


@pytest.mark.asyncio
@pytest.mark.timeout(20)
@pytest.mark.parametrize("step", ["remove_unselected_files", "remove_lfs_pointers"])
async def test_a_cancelled_checkout_cleanup_finishes_before_the_lock_is_released(
    origin, clones, monkeypatch, step
):
    """These steps delete files in a worker thread, and a cancelled to_thread await does not
    stop it: the resolver must keep the clone lock until the thread is done, or the thread
    would delete files from the clone the next run publishes."""
    import asyncio
    import threading

    selection = importlib.import_module("cognee.tasks.code_graph.repo_selection")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = getattr(selection, step)

    def paused(*args):
        entered.set()
        try:
            assert release.wait(10), "Test did not release the cleanup step"
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(selection, step, paused)
    task = asyncio.create_task(resolve_module.resolve_repo_source(_URL, clones_dir=clones))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "Resolver abandoned a live filesystem writer"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_path_filter_change_same_commit_refreshes_sparse_snapshot(origin, clones):
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    (origin.work / "one.py").write_text("one = 1\n")
    (origin.work / "two.py").write_text("two = 2\n")
    origin.commit("main", "selected paths")
    target = await resolve_module.resolve_repo_source(
        _URL, clones_dir=clones, include_paths=["one.py"]
    )
    first = read_repository_snapshot(target)
    assert (target / "one.py").is_file()
    assert not (target / "two.py").exists()
    second_blob = git_output(origin.work, "rev-parse", "HEAD:two.py")
    assert second_blob not in local_objects(target)

    await resolve_module.resolve_repo_source(_URL, clones_dir=clones, include_paths=["two.py"])
    second = read_repository_snapshot(target)
    assert not (target / "one.py").exists()
    assert (target / "two.py").is_file()
    assert first["commit"] == second["commit"]
    assert first["selection_fingerprint"] != second["selection_fingerprint"]

    await resolve_module.resolve_repo_source(_URL, clones_dir=clones, include_paths=[])
    assert not (target / "two.py").exists()
    assert read_repository_snapshot(target)["selected_files"] == 0


@pytest.mark.asyncio
async def test_empty_path_selection_sweeps_previously_extracted_graph(origin, clones, monkeypatch):
    """Real Git + Enola: an empty selection is a changed graph, not an incremental skip."""
    from cognee.tasks.code_graph.enola import EnolaNotInstalledError, find_enola_binary
    from cognee.tasks.code_graph.extract_code_graph import add_code_graph_edges, extract_code_graph

    try:
        find_enola_binary()
    except EnolaNotInstalledError:
        pytest.skip("enola is required")
    graph_module = importlib.import_module("cognee.infrastructure.databases.graph.get_graph_engine")
    engine = AsyncMock()
    engine.get_node.return_value = None
    engine.get_graph_data.return_value = ([], [])
    monkeypatch.setattr(graph_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        importlib.import_module("cognee.infrastructure.databases.graph"),
        "get_graph_engine",
        AsyncMock(return_value=engine),
    )
    (origin.work / "one.py").write_text("def hello():\n    return 1\n")
    origin.commit("main", "code to remove")
    repo = await resolve_module.resolve_repo_source(
        _URL, clones_dir=clones, include_paths=["one.py"]
    )
    namespace = "github.com/org/partial"
    initial = await extract_code_graph(repo_path=repo, repo_name=namespace, namespace=namespace)
    old_ids = {str(point.id) for point in initial if point.__class__.__name__ != "CodeRepository"}
    assert old_ids
    engine.get_graph_data.return_value = (
        [
            (str(point.id), {**point.model_dump(), "type": point.__class__.__name__})
            for point in initial
        ],
        [],
    )
    await resolve_module.resolve_repo_source(_URL, clones_dir=clones, include_paths=[])
    emptied = await extract_code_graph(repo_path=repo, repo_name=namespace, namespace=namespace)
    assert emptied  # [] is the incremental no-op signal; empty content must not produce it.
    await add_code_graph_edges(emptied, repo_path=repo, repo_name=namespace, namespace=namespace)
    removed_ids = {
        node_id for call in engine.delete_nodes.await_args_list for node_id in call.args[0]
    }
    assert old_ids <= removed_ids
