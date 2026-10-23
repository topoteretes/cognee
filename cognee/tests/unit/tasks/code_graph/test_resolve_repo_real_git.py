"""resolve_repo_source against real git and a local origin.

A bare branch name is resolved by git itself (a tag before a branch, ``+x`` as a force
refspec, ``tag`` as a keyword), so a refresh that fetched it unqualified moved the clone
to another commit than ``clone --branch`` had picked. The repository URL is mapped to a
bare repository on disk, so nothing here touches the network.
"""

import asyncio
import importlib
import os
import shutil
import subprocess
import threading
from unittest.mock import AsyncMock

import pytest
from filelock import FileLock, Timeout

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
ssrf_module = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")

_URL = "https://github.com/org/repo"
_IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(cwd, *args):
    subprocess.run(
        ["git", *args], cwd=cwd, env={**os.environ, **_IDENTITY}, check=True, capture_output=True
    )


class Origin:
    """A bare repository with a work tree to push from; every commit sets BRANCH.txt."""

    def __init__(self, root):
        root.mkdir(parents=True)
        self.bare = root / "origin.git"
        self.work = root / "work"
        _git(root, "init", "-q", "--bare", "-b", "main", str(self.bare))
        _git(self.bare, "config", "uploadpack.allowFilter", "true")
        _git(root, "init", "-q", "-b", "main", str(self.work))
        _git(self.work, "remote", "add", "origin", str(self.bare))
        self._commit("main-1")
        _git(self.work, "push", "-q", "origin", "main")

    def _commit(self, text):
        (self.work / "BRANCH.txt").write_text(text)
        _git(self.work, "add", "-A")
        _git(self.work, "commit", "-q", "-m", text)

    def commit(self, branch, text):
        """Commit on ``branch`` (created from main when new) and force-push it."""
        known = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"refs/heads/{branch}"],
            cwd=self.work,
            capture_output=True,
            check=False,
        )
        if known.returncode == 0:
            _git(self.work, "switch", "-q", branch)
        else:
            _git(self.work, "switch", "-q", "-c", branch, "main")
        self._commit(text)
        # Spelled out: a bare 'rel', 'tag' or '+dev' is exactly what git misreads.
        _git(self.work, "push", "-q", "-f", "origin", f"refs/heads/{branch}:refs/heads/{branch}")

    def rewrite(self, branch, text):
        """Replace the tip of ``branch`` with another commit and force-push it."""
        _git(self.work, "switch", "-q", branch)
        _git(self.work, "reset", "-q", "--hard", "HEAD~1")
        self.commit(branch, text)

    def tag(self, name, text, move=False):
        """Push a tag on a commit that no branch holds; ``move`` re-points an existing tag."""
        flags = ["-f"] if move else []
        _git(self.work, "switch", "-q", "-c", "scratch", "main")
        self._commit(text)
        _git(self.work, "tag", *flags, name)
        _git(self.work, "push", "-q", *flags, "origin", f"refs/tags/{name}:refs/tags/{name}")
        _git(self.work, "switch", "-q", "main")
        _git(self.work, "branch", "-q", "-D", "scratch")

    def delete(self, branch):
        _git(self.work, "push", "-q", "origin", f":refs/heads/{branch}")


@pytest.fixture(autouse=True)
def _hermetic_git(tmp_path, monkeypatch):
    """Run every git command, the test's and resolve_repo's, on a bare configuration.

    The developer's global config (``commit.gpgsign`` with a signer that is not running,
    ``core.hooksPath``, ``init.templateDir``) would fail the commits made here, and a test
    run from a git hook would act on that hook's repository through GIT_DIR.
    """
    empty_config = tmp_path / "gitconfig"  # stands in for /dev/null, absent on Windows
    empty_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_TEMPLATE_DIR", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    origin = Origin(tmp_path / "remote")
    real_exec_git = resolve_module._exec_git

    async def exec_git(args, cwd=None, env=None):
        # A file:// URL, not a path: only then is the clone shallow, like a real remote.
        remote = origin.bare.resolve().as_uri()
        return await real_exec_git([remote if arg == _URL else arg for arg in args], cwd, env)

    monkeypatch.setattr(resolve_module, "_exec_git", exec_git)
    real_origin_url = resolve_module._origin_url

    async def origin_url(clone):
        # The clone's origin is the file:// URL the repository URL was mapped to.
        stored = await real_origin_url(clone)
        return _URL if stored == origin.bare.resolve().as_uri() else stored

    monkeypatch.setattr(resolve_module, "_origin_url", origin_url)
    monkeypatch.setattr(ssrf_module, "validate_outbound_url", AsyncMock())
    return origin


@pytest.fixture
def clones(tmp_path):
    return tmp_path / "clones"


async def _resolve(clones, branch=None):
    return await resolve_module.resolve_repo_source(_URL, clones_dir=clones, branch=branch)


def _content(path):
    return (path / "BRANCH.txt").read_text().strip()


def _clones_in(clones):
    """The clone directories, without the lock directory and temporary clones beside them."""
    return [path for path in clones.iterdir() if not path.name.startswith(".")]


def _temporary_clones(clones):
    return [path for path in clones.iterdir() if ".tmp-" in path.name]


# Each name is one git misreads when fetched bare.
_NAMESAKES = {
    "rel": lambda origin: origin.tag("rel", "rel-tag"),  # a tag is read before the branch
    "+dev": lambda origin: origin.commit("dev", "dev-1"),  # a force refspec for 'dev'
    "tag": lambda origin: None,  # a keyword that wants a tag name
}


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", _NAMESAKES)
async def test_a_refresh_follows_the_branch_and_not_a_namesake(origin, clones, branch):
    _NAMESAKES[branch](origin)
    origin.commit(branch, f"{branch}-1")
    assert _content(await _resolve(clones, branch)) == f"{branch}-1"

    origin.commit(branch, f"{branch}-2")

    assert _content(await _resolve(clones, branch)) == f"{branch}-2"


@pytest.mark.asyncio
async def test_a_tag_is_not_taken_for_a_branch(origin, clones):
    origin.tag("v1", "v1-a")  # 'clone --branch v1' would take it and leave HEAD detached

    with pytest.raises(resolve_module.CodeRepositoryError, match="Branch 'v1' was not found"):
        await _resolve(clones, "v1")

    assert _clones_in(clones) == []


@pytest.mark.asyncio
async def test_each_branch_is_a_clone_of_its_own_that_follows_force_pushes(origin, clones):
    origin.commit("dev", "dev-1")
    origin.commit("feature/ü", "ü-1")

    dev = await _resolve(clones, "dev")
    feature = await _resolve(clones, "feature/ü")
    default = await _resolve(clones)

    assert [_content(path) for path in (dev, feature, default)] == ["dev-1", "ü-1", "main-1"]
    assert default == resolve_module.clone_dir(_URL, clones_dir=clones)
    assert sorted(path.name for path in _clones_in(clones)) == sorted(
        path.name for path in (dev, feature, default)
    )

    origin.commit("feature/ü", "ü-2")
    assert _content(await _resolve(clones, "feature/ü")) == "ü-2"
    origin.rewrite("feature/ü", "ü-3")  # a force push is not a fast-forward
    assert _content(await _resolve(clones, "feature/ü")) == "ü-3"
    assert _content(dev) == "dev-1"  # the other branches' clones are never touched
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=feature,
        capture_output=True,
        text=True,
        check=True,
    )
    assert shallow.stdout.strip() == "true"


@pytest.mark.asyncio
async def test_a_missing_branch_is_reported_and_never_replaces_a_clone(origin, clones):
    with pytest.raises(resolve_module.CodeRepositoryError, match="Branch 'nope' was not found"):
        await _resolve(clones, "nope")
    assert _clones_in(clones) == []

    origin.commit("dev", "dev-1")
    dev = await _resolve(clones, "dev")
    origin.delete("dev")
    origin.tag("dev", "dev-tag")  # a bare 'dev' would still match, and hide the deletion

    with pytest.raises(resolve_module.CodeRepositoryError, match="Branch 'dev' was not found"):
        await _resolve(clones, "dev")
    assert _content(dev) == "dev-1"


@pytest.mark.asyncio
async def test_the_origin_is_read_as_stored(tmp_path):
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", str(repo))
    assert await resolve_module._origin_url(repo) is None

    _git(repo, "remote", "add", "origin", "https://github.com/org/repo")
    _git(repo, "config", "url.https://mirror.example/.insteadOf", "https://github.com/")

    assert await resolve_module._origin_url(repo) == "https://github.com/org/repo"


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_clone(origin, clones, monkeypatch):
    run_git = resolve_module._run_git
    clone_calls = []

    async def counting_run_git(args, cwd=None, env=None):
        if args[0] == "clone":
            clone_calls.append(args)
        return await run_git(args, cwd, env)

    monkeypatch.setattr(resolve_module, "_run_git", counting_run_git)

    resolved = await asyncio.gather(*(_resolve(clones) for _ in range(4)))

    # The lock lets one caller clone; the others find its finished clone and refresh it.
    assert len(clone_calls) == 1
    assert len(set(resolved)) == 1 and _content(resolved[0]) == "main-1"
    assert _temporary_clones(clones) == []


@pytest.mark.asyncio
async def test_a_clone_is_complete_before_it_appears_at_its_target(origin, clones, monkeypatch):
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    run_git = resolve_module._run_git
    seen = []

    async def watching_run_git(args, cwd=None, env=None):
        result = await run_git(args, cwd, env)
        if args[0] == "clone":
            seen.append((target.exists(), resolve_module.Path(args[-1]).parent == clones))
        return result

    monkeypatch.setattr(resolve_module, "_run_git", watching_run_git)

    assert await _resolve(clones) == target

    # Cloned beside the target, which appears only once the clone is done.
    assert seen == [(False, True)]
    assert _content(target) == "main-1"


@pytest.mark.asyncio
async def test_an_interrupted_clone_is_cloned_again_not_returned(origin, clones):
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    # What a clone killed before its first commit arrived leaves: a .git with an origin
    # and no commit.
    _git(clones.parent, "init", "-q", str(target))
    _git(target, "remote", "add", "origin", origin.bare.resolve().as_uri())

    assert await _resolve(clones) == target
    assert _content(target) == "main-1"


@pytest.mark.asyncio
async def test_temporary_clones_are_swept_and_a_failed_clone_leaves_nothing(origin, clones):
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    leftover = clones / f".{target.name}.tmp-dead"
    (leftover / ".git").mkdir(parents=True)  # a holder that died mid-clone
    shutil.rmtree(origin.bare)

    with pytest.raises(resolve_module.CodeRepositoryError, match="Failed to clone"):
        await _resolve(clones)

    assert _temporary_clones(clones) == []
    assert _clones_in(clones) == []


@pytest.mark.asyncio
async def test_the_users_ssh_command_is_kept_in_batch_mode(tmp_path, monkeypatch):
    for name in ("GIT_SSH_COMMAND", "GIT_SSH"):
        monkeypatch.delenv(name, raising=False)
    # _hermetic_git points the global config at this file.
    (tmp_path / "gitconfig").write_text("[core]\n\tsshCommand = ssh -i /keys/deploy\n")
    assert await resolve_module._configured_ssh_command() == "ssh -i /keys/deploy"
    seen = []

    async def failing_clone(args, cwd=None, env=None):
        seen.append(env)
        return 128, "fatal: unreachable"

    monkeypatch.setattr(resolve_module, "_run_git", failing_clone)

    with pytest.raises(resolve_module.CodeRepositoryError):
        await resolve_module.resolve_repo_source(
            "git@github.com:org/repo.git", clones_dir=tmp_path / "clones"
        )
    assert seen[0]["GIT_SSH_COMMAND"] == "ssh -i /keys/deploy -o BatchMode=yes"


@pytest.mark.asyncio
async def test_default_branch_refresh_follows_force_push(origin, clones):
    origin.commit("main", "main-2")
    target = await _resolve(clones)
    origin.rewrite("main", "main-rewritten")

    assert await _resolve(clones) == target
    assert _content(target) == "main-rewritten"


@pytest.mark.asyncio
async def test_default_branch_refresh_follows_remote_head_change(origin, clones):
    target = await _resolve(clones)
    origin.commit("new-default", "new-default-1")
    _git(origin.bare, "symbolic-ref", "HEAD", "refs/heads/new-default")
    origin.delete("main")

    assert await _resolve(clones) == target
    assert _content(target) == "new-default-1"
    origin.commit("new-default", "new-default-2")
    assert _content(await _resolve(clones)) == "new-default-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", [None, "main"])
async def test_unreachable_origin_never_returns_stale_success(origin, clones, branch):
    target = await _resolve(clones, branch)
    shutil.rmtree(origin.bare)

    with pytest.raises(resolve_module.CodeRepositoryError, match="Could not refresh"):
        await _resolve(clones, branch)

    assert _content(target) == "main-1"


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_cleanup_before_releasing_lock(
    origin, clones, monkeypatch
):
    target = resolve_module.clone_dir(_URL, clones_dir=clones)
    clone_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = threading.Event()
    run_git = resolve_module._run_git
    rmtree = shutil.rmtree
    loop = asyncio.get_running_loop()

    async def interrupted_clone(args, cwd=None, env=None):
        if args[0] == "clone":
            (resolve_module.Path(args[-1]) / "partial-file").write_text("incomplete")
            clone_started.set()
            await asyncio.Event().wait()
        return await run_git(args, cwd=cwd, env=env)

    def slow_cleanup(path, *args, **kwargs):
        loop.call_soon_threadsafe(cleanup_started.set)
        assert allow_cleanup.wait(timeout=5)
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(resolve_module, "_run_git", interrupted_clone)
    monkeypatch.setattr(resolve_module.shutil, "rmtree", slow_cleanup)
    task = asyncio.create_task(_resolve(clones))
    try:
        await asyncio.wait_for(clone_started.wait(), timeout=5)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "Cancellation must wait for filesystem cleanup"
        with (
            pytest.raises(Timeout),
            FileLock(str(clones / ".locks" / f"{target.name}.lock"), timeout=0),
        ):
            pytest.fail("Lock released while cleanup was still running")
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not target.exists()
        assert _temporary_clones(clones) == []
    finally:
        allow_cleanup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(resolve_module, "_run_git", run_git)
    monkeypatch.setattr(resolve_module.shutil, "rmtree", rmtree)
    assert _content(await _resolve(clones)) == "main-1"


@pytest.mark.asyncio
async def test_a_repository_whose_name_is_taken_is_refused_and_leaves_no_clone(origin, clones):
    """'org/repo' and 'org-repo' read as one name; the second one in is refused."""
    await resolve_module._claim_graph_name(clones, "https://github.com/org-repo")

    with pytest.raises(resolve_module.CodeRepositoryError, match="already used by"):
        await _resolve(clones)

    assert _clones_in(clones) == []
    assert _temporary_clones(clones) == []


@pytest.mark.asyncio
async def test_a_clone_that_fails_does_not_take_the_name(origin, clones):
    with pytest.raises(resolve_module.CodeRepositoryError, match="was not found"):
        await _resolve(clones, "no-such-branch")

    assert not (clones / ".names").exists() or not any((clones / ".names").iterdir())
    await resolve_module._claim_graph_name(clones, "https://github.com/org-repo")


@pytest.mark.asyncio
async def test_a_failed_clone_does_not_show_the_server_directory(origin, clones, monkeypatch):
    async def failing_run_git(args, cwd=None, env=None):
        return 128, f"fatal: Cloning into '{args[-1]}' failed"

    monkeypatch.setattr(resolve_module, "_run_git", failing_run_git)

    with pytest.raises(resolve_module.CodeRepositoryError) as error:
        await _resolve(clones)

    assert str(clones) not in error.value.message
    assert str(clones.resolve()) not in error.value.message


@pytest.mark.asyncio
async def test_partial_clone_omits_large_binary_and_pins_commit(origin, clones, monkeypatch):
    from cognee.tasks.code_graph.repo_resources import disk_usage_bytes
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    (origin.work / "weights.bin").write_bytes(os.urandom(8 * 1024 * 1024))
    (origin.work / "app.py").write_text("print('hello')")
    origin.commit("main", "with-artifact")
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", str(1024 * 1024))

    target = await _resolve(clones)

    assert (target / "app.py").read_text() == "print('hello')"
    assert not (target / "weights.bin").exists()
    assert disk_usage_bytes(clones) < 1024 * 1024
    snapshot = read_repository_snapshot(target)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    assert snapshot["commit"] == head
    assert snapshot["skipped"]["binary"] == 1
    # The binary is not merely absent from the checkout: it never arrived in .git.
    blob = subprocess.check_output(
        ["git", "rev-parse", "HEAD:weights.bin"], cwd=origin.work, text=True
    ).strip()
    local_objects = subprocess.check_output(
        ["git", "cat-file", "--batch-all-objects", "--batch-check=%(objectname)"],
        cwd=target,
        text=True,
    ).splitlines()
    assert blob not in local_objects


@pytest.mark.asyncio
async def test_document_assets_follow_the_requested_mode_on_refresh(origin, clones):
    (origin.work / "architecture.png").write_bytes(b"image-document")
    (origin.work / "README.md").write_text("# architecture")
    origin.commit("main", "with-documents")

    target = await resolve_module.resolve_repo_source(
        _URL, clones_dir=clones, include_documents=False
    )
    assert not (target / "architecture.png").exists()
    assert (target / "README.md").exists()  # enola's deterministic markdown extractor

    assert await _resolve(clones) == target
    assert (target / "architecture.png").read_bytes() == b"image-document"
    await resolve_module.resolve_repo_source(_URL, clones_dir=clones, include_documents=False)
    assert not (target / "architecture.png").exists()


@pytest.mark.asyncio
async def test_hidden_configuration_and_custom_extractor_inputs_are_preserved(origin, clones):
    files = {
        ".gitignore": "generated/\n",
        ".github/workflows/check.yml": "name: check\n",
        "custom/input.unusual": "custom extractor input",
    }
    for name, content in files.items():
        path = origin.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    origin.commit("main", "extractor-config")

    target = await resolve_module.resolve_repo_source(
        _URL, clones_dir=clones, include_documents=False
    )
    for name, content in files.items():
        assert (target / name).read_text() == content


@pytest.mark.asyncio
async def test_a_server_ignoring_the_filter_is_stopped_and_leaves_no_clone(origin, clones):
    _git(origin.bare, "config", "uploadpack.allowFilter", "false")

    with pytest.raises(resolve_module.CodeRepositoryError, match="ignored the partial clone"):
        await _resolve(clones)

    assert _clones_in(clones) == []
    assert _temporary_clones(clones) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_disk_budget_failure_cleans_new_clone_and_failed_refresh(
    origin, clones, monkeypatch, existing
):
    from cognee.tasks.code_graph.repo_resources import RepositoryResourceError

    if existing:
        await _resolve(clones)
    (origin.work / "large.py").write_bytes(os.urandom(2 * 1024 * 1024))
    origin.commit("main", "large-source")
    monkeypatch.setenv("COGNEE_REPOS_MAX_BYTES", str(1024 * 1024))

    with pytest.raises(RepositoryResourceError):
        await _resolve(clones)

    assert _temporary_clones(clones) == []
    assert not resolve_module.clone_dir(_URL, clones_dir=clones).exists()


@pytest.mark.asyncio
async def test_exact_sparse_paths_keep_spaces_and_glob_characters(origin, clones):
    names = ["space name.py", "[literal].py", "star*.py", "question?.py", "trailing .py"]
    if os.name == "nt":
        names = names[:2]  # Windows cannot represent '*' or '?' in a filename.
    for name in names:
        (origin.work / name).write_text("# " + name)
    origin.commit("main", "literal-paths")

    target = await _resolve(clones)
    for name in names:
        assert (target / name).read_text() == "# " + name


@pytest.mark.asyncio
async def test_selected_lfs_pointer_fails_without_running_smudge(origin, clones, tmp_path):
    # A configured smudge filter would write this marker if checkout invoked it.
    marker = tmp_path / "smudged"
    _git(origin.work, "config", "filter.lfs.smudge", f"touch {marker}")
    (origin.work / ".gitattributes").write_text("large.py filter=lfs\n")
    (origin.work / "large.py").write_text(
        "version https://git-lfs.github.com/spec/v1\n" + "oid sha256:" + "a" * 64 + "\nsize 123\n"
    )
    origin.commit("main", "lfs-pointer")
    # The config belongs to the Git client, not the clone's remote.
    _git(origin.work, "config", "--global", "filter.lfs.smudge", f"touch {marker}")
    _git(origin.work, "config", "--global", "filter.lfs.required", "true")

    with pytest.raises(resolve_module.CodeRepositoryError, match="Git LFS"):
        await _resolve(clones)
    assert not marker.exists()
    assert _temporary_clones(clones) == []


@pytest.mark.asyncio
async def test_submodules_are_recorded_without_recursive_clone(origin, clones):
    from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin.work, text=True
    ).strip()
    _git(origin.work, "update-index", "--add", "--cacheinfo", f"160000,{commit},dependency")
    _git(origin.work, "commit", "-qm", "submodule")
    _git(origin.work, "push", "-q", "origin", "main")
    target = await _resolve(clones)
    assert read_repository_snapshot(target)["skipped"]["submodule"] == 1
    assert not (target / "dependency" / ".git").exists()


@pytest.mark.asyncio
async def test_incomplete_tree_never_checks_out_or_publishes_clone(origin, clones, monkeypatch):
    real_exec = resolve_module._exec_git
    calls = []

    async def truncated_tree(args, cwd=None, env=None):
        calls.append(args[0])
        code, stdout, stderr = await real_exec(args, cwd=cwd, env=env)
        return code, stdout.rstrip("\0") if args[0] == "ls-tree" else stdout, stderr

    monkeypatch.setattr(resolve_module, "_exec_git", truncated_tree)
    with pytest.raises(resolve_module.CodeRepositoryError, match="tree listing is incomplete"):
        await _resolve(clones)
    assert "reset" not in calls
    assert not resolve_module.clone_dir(_URL, clones_dir=clones).exists()
    assert _temporary_clones(clones) == []
