"""Resolve repository specs (local paths or remote git URLs) to local clones.

Used by ``remember(content_type="code")`` to resolve repository specs and by
``add()`` to recognise a GitHub/GitLab repository URL
(:func:`code_repo_clone_url`) and shallow-clone it for the enola code-graph
route, and by the GitHub integration sync to clone private repositories with
an installation token (``resolve_repo_source(credentials=...)``). Clones live under
``BaseConfig.repos_root_directory`` (``~/.cognee/repos`` by default) and are
reused across calls once its ``origin`` is checked; an existing clone is
refreshed by fetching and resetting to the remote tip; a failed refresh raises.
A ``branch`` can be chosen for a remote
repository; each branch is cloned into a directory of its own. A new clone is
made in a temporary sibling directory and renamed into place once complete.
Cloning and refreshing one clone directory are serialized by a file lock, across
processes; reading the clone afterwards (building its code graph, adding its
documents) runs outside that lock, so a refresh by another caller can change
the clone while it is read: route the indexing of one repository URL to one
worker. Two repositories whose URLs differ only in separators ('a-b/c' and
'a/b-c') share one readable name, which names their code graph and manifest
row; the first one indexed keeps it and the other is refused
(:func:`_claim_graph_name`). Clones made before directory names carried a hash
(``<clones>/<readable name>``) are not reused: each repository is cloned again
into its hashed directory, and the old directory is left in place.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import status

from cognee.base_config import get_base_config
from cognee.exceptions import CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("code_graph")

_REMOTE_PREFIXES = ("https://", "http://", "git@", "ssh://")
# Written into .git of a clone made with credentials. Such a clone may hold a
# private repository, so it is handed out only after a refresh that proves the
# caller's access: a failed refresh is refused instead of handing over the stale
# copy, whatever credentials the caller passed (they may be wrong).
_CREDENTIALED_MARKER = "cognee-credentialed"

# Per-clone lock files, in a directory of their own inside the clones directory.
_LOCKS_DIR = ".locks"
# Which repository owns each readable name, one file per name (see _claim_graph_name).
_NAMES_DIR = ".names"

_FALSEY = {"false", "0", "no", "off"}

_GIT_TIMEOUT_SECONDS = 600
_FILTER_REFUSED = b"filtering not recognized by server"

# github.com / gitlab.com repository roots are detected by shape; any other
# host is a repository only when the URL ends in ".git". These are the
# two-segment forge paths that are site pages, not <owner>/<repo>.
_GITHUB_RESERVED_OWNERS = frozenset(
    {
        "about",
        "apps",
        "collections",
        "enterprise",
        "explore",
        "features",
        "login",
        "marketplace",
        "notifications",
        "organizations",
        "orgs",
        "pricing",
        "search",
        "security",
        "settings",
        "site",
        "sponsors",
        "topics",
        "trending",
        "users",
    }
)
_GITLAB_RESERVED_GROUPS = frozenset(
    {
        "admin",
        "api",
        "dashboard",
        "explore",
        "groups",
        "help",
        "oauth",
        "profile",
        "projects",
        "search",
        "users",
    }
)


def default_clones_dir() -> Path:
    """Where remote repositories are cloned: ``BaseConfig.repos_root_directory``.

    ``COGNEE_REPOS_DIR``, default ``~/.cognee/repos``. Read at call time. The
    same directory is one of ingestion's always-allowed local roots, so the
    document files of a clone can be added by path.
    """
    return Path(get_base_config().repos_root_directory)


class CodeRepositoryError(CogneeSystemError):
    def __init__(
        self,
        message: str = "Could not resolve the code repository.",
        name: str = "CodeRepositoryError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


def is_remote_repo(spec) -> bool:
    """Whether the spec is a remote git URL rather than a local path."""
    return isinstance(spec, str) and spec.startswith(_REMOTE_PREFIXES)


def code_repo_clone_url(spec) -> str | None:
    """The clone URL when an http(s) string names a whole git repository, else None.

    This is what lets ``add()`` treat ``https://github.com/<owner>/<repo>`` as a
    code project instead of a web page to scrape. Detected: repository roots on
    github.com (``<owner>/<repo>``) and gitlab.com (``<group>/<repo>``, nested
    groups included), with an optional ``.git`` suffix, trailing slash, ``www.``
    prefix, query or fragment -- and any http(s) URL ending in ``.git`` on any
    host. Deeper forge URLs (``/blob/``, ``/tree/``, ``/issues``, ``/pull/``,
    GitLab's ``/-/`` pages) and forge site pages (``/topics/x``, ``/explore``)
    name pages, not repositories, and are left to the web-page path. ``git@``
    and ``ssh://`` specs are not detected: clone them yourself and pass the
    local directory.

    The returned URL is normalised for ``git clone``: query and fragment
    dropped, trailing slash removed, userinfo (a token) kept -- redact it with
    :func:`redact_repo_spec` before persisting or logging.
    """
    if not isinstance(spec, str):
        return None
    parts = urlsplit(spec.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    host = host.removeprefix("www.")
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 2:
        return None

    if segments[-1].endswith(".git"):
        is_repo = True
    elif host == "github.com":
        is_repo = len(segments) == 2 and segments[0] not in _GITHUB_RESERVED_OWNERS
    elif host == "gitlab.com":
        is_repo = "-" not in segments and segments[0] not in _GITLAB_RESERVED_GROUPS
    else:
        is_repo = False
    if not is_repo:
        return None
    return urlunsplit((parts.scheme, parts.netloc, "/" + "/".join(segments), "", ""))


def redact_repo_spec(spec: str | Path) -> str:
    """The spec with any URL-embedded credentials removed.

    Connectors pass short-lived tokens in the URL userinfo
    (``https://x-access-token:<token>@github.com/...``); anything persisted
    or logged — clone slugs, the stored git remote, result items, error
    messages — must use this form instead of the raw spec.
    """
    url = str(spec)
    if "://" not in url:
        return url
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _repo_key(url: str) -> str:
    """The part of a credential-free remote URL that names the repository.

    The scheme, a trailing slash and a ``.git`` suffix are dropped, so the
    spellings of one repository share a clone and pass its origin check.
    """
    return url.split("://")[-1].rstrip("/").removesuffix(".git")


def _readable_slug(url: str) -> str:
    """The readable directory name of a remote URL, e.g. 'github.com-org-repo'.

    Every run of characters other than letters, digits, '.', '_' and '-' becomes one '-'.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", _repo_key(url).replace(":", "/")).strip("-.")


def _clone_slug(url: str, branch: str | None = None) -> str:
    """A stable directory name for a remote URL, e.g. 'github.com-org-repo-<hash>'.

    The readable part merges every run of other characters into one '-', so
    'github.com/a-b/c' and 'github.com/a/b-c' read the same; the hash of the
    repository's exact name (:func:`_repo_key`) keeps their clones apart.

    A chosen branch gets its own directory beside it, e.g.
    'github.com-org-repo-<hash>@dev-<hash>', so no clone is ever replaced under another
    caller. The hash of the exact name keeps apart branches the readable part merges:
    case-insensitive file systems, and the characters replaced here.
    """
    key = _repo_key(url)
    slug = f"{_readable_slug(url)}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"
    if branch is None:
        return slug
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", branch)[:40]
    return f"{slug}@{readable}-{hashlib.sha256(branch.encode()).hexdigest()[:8]}"


def clone_dir(url: str, branch: str | None = None, clones_dir: Path | None = None) -> Path:
    """Where :func:`resolve_repo_source` clones the credential-free ``url`` on ``branch``
    (None: the remote's default branch, in a directory of its own)."""
    return (Path(clones_dir) if clones_dir else default_clones_dir()) / _clone_slug(url, branch)


def repo_identity(url: str) -> Path:
    """The Data-row identity of the credential-free remote ``url``.

    The path its default-branch clone had before clone directories carried a
    hash (:func:`_clone_slug`). Rows are keyed on it, so repositories already
    stored keep their row: keying on the hashed directory would add a second
    manifest row for each of them on the next add. Not a directory anything is
    cloned into.
    """
    return default_clones_dir() / _readable_slug(url)


def clone_repo_name(url: str, clone: str | Path) -> str:
    """The repository name the code graph gives ``clone``, a clone of the credential-free ``url``.

    enola names a repository after its directory, and every code node id derives
    from that name. Clone directories now carry a hash (:func:`_clone_slug`), so
    the name is the one the directory had before: the readable slug, followed by
    a branch clone's own suffix. A repository already stored keeps its node ids,
    where the hashed name would add a second subgraph that nothing sweeps. A
    directory that is not a clone of ``url`` keeps its own name.
    """
    name = Path(clone).name
    hashed = _clone_slug(url)
    if name == hashed or name.startswith(hashed + "@"):
        return _readable_slug(url) + name[len(hashed) :]
    return name


# git's refname rules that cannot appear anywhere in a branch name: control
# characters, space, ~ ^ : ? * [ and backslash.
_BRANCH_FORBIDDEN_CHARS = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")


def _invalid_branch_reason(branch) -> str | None:
    """Why ``branch`` cannot be a branch name, or None: ``git check-ref-format --branch``
    semantics (checked on git 2.50.1; a bare ``@`` is refused on purpose), so a name git
    would read as an option never reaches it."""
    if not isinstance(branch, str) or not branch:
        return "it must be a non-empty string"
    if branch.startswith("-"):
        return "it must not start with '-'"
    if branch in ("HEAD", "@"):
        return f"'{branch}' is not a branch name"
    if _BRANCH_FORBIDDEN_CHARS.search(branch):
        return "it contains a space, control character or one of ~ ^ : ? * [ \\"
    if ".." in branch or "@{" in branch or "//" in branch:
        return "it contains '..', '@{' or '//'"
    if branch.startswith("/") or branch.endswith(("/", ".")):
        return "it must not start with '/' or end with '/' or '.'"
    if any(part.startswith(".") or part.endswith(".lock") for part in branch.split("/")):
        return "a path component must not start with '.' or end with '.lock'"
    return None


def check_branch_name(branch) -> None:
    """Raise ``ValueError`` unless ``branch`` is a usable branch name."""
    reason = _invalid_branch_reason(branch)
    if reason is not None:
        raise ValueError(f"Invalid branch name {branch!r}: {reason}.")


def _credential_env(token: str, url: str) -> dict:
    """Auth for git over https, injected as environment-level git config.

    The environment channel (git >= 2.31) keeps the token out of argv (which
    any local process can read from the process listing), out of every
    URL-shaped string (so nothing derived from the URL can leak it into
    logs or error messages), and out of the on-disk remote config.

    The header is scoped to the origin of the credential-free ``url``
    (``http.<scheme>://<host>/.extraHeader``), so git sends it to that host
    only, never to another host a redirect or a submodule points at. A remote
    that is not http(s) gets nothing: the header is http config.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        return {}
    host = parts.netloc.rsplit("@", 1)[-1].lower()
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"http.{scheme}://{host}/.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
    }


def _noninteractive_env(ssh_command: str | None = None) -> dict:
    """Git never waits for input: no username/password prompt over https, and
    ssh in batch mode (no passphrase or host-key prompt). ``ssh_command`` is the
    user's ``core.sshCommand`` (see :func:`_configured_ssh_command`), run in
    batch mode in place of plain ssh. A user's own GIT_SSH_COMMAND or GIT_SSH
    is left in charge of ssh."""
    env = {"GIT_TERMINAL_PROMPT": "0"}
    if "GIT_SSH_COMMAND" not in os.environ and "GIT_SSH" not in os.environ:
        env["GIT_SSH_COMMAND"] = f"{ssh_command or 'ssh'} -o BatchMode=yes"
    return env


# Git config every invocation carries: no credential helper and no askpass
# program, so git authenticates only with what the caller passed (the
# environment-level header of _credential_env, or URL userinfo). Without these a
# refresh of a private clone could succeed on the server's stored credentials (a
# helper, a keychain) for a caller who brought none, and unlock that clone. An
# empty credential.helper resets the helper list built from the system, global
# and repository config, which git reads before the environment's entries.
_NO_STORED_CREDENTIALS_CONFIG = (("credential.helper", ""), ("core.askPass", ""))
_NO_LFS_SMUDGE_CONFIG = (
    ("filter.lfs.process", ""),
    ("filter.lfs.smudge", ""),
    ("filter.lfs.required", "false"),
)


def _git_config_entries(env) -> list[tuple[str, str]]:
    """The ``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_<n>`` / ``GIT_CONFIG_VALUE_<n>`` entries of ``env``."""
    try:
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        count = 0
    return [
        (env.get(f"GIT_CONFIG_KEY_{index}", ""), env.get(f"GIT_CONFIG_VALUE_{index}", ""))
        for index in range(count)
    ]


def _is_git_config_entry(name: str) -> bool:
    return name == "GIT_CONFIG_COUNT" or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))


def _git_env(env: dict | None = None) -> dict:
    """The environment git runs with: the process's, non-interactive, ``env`` on top,
    and no stored credentials (see :data:`_NO_STORED_CREDENTIALS_CONFIG`).

    Environment-level config entries are renumbered into one list: the process's own
    first, then ``env``'s (e.g. :func:`_credential_env`), then the hardening entries
    last, so they win and no index of one source overwrites another's. GIT_ASKPASS
    and SSH_ASKPASS are set empty: git then skips every askpass program, and with
    GIT_TERMINAL_PROMPT=0 a missing credential fails instead of being asked for.
    """
    extra = env or {}
    entries = (
        _git_config_entries(os.environ)
        + _git_config_entries(extra)
        + list(_NO_STORED_CREDENTIALS_CONFIG)
        + list(_NO_LFS_SMUDGE_CONFIG)
    )
    merged = {
        name: value
        for source in (os.environ, _noninteractive_env(), extra)
        for name, value in source.items()
        if not _is_git_config_entry(name)
    }
    merged["GIT_ASKPASS"] = ""
    merged["SSH_ASKPASS"] = ""
    # LFS contents are never downloaded implicitly during a sparse checkout.
    merged["GIT_LFS_SKIP_SMUDGE"] = "1"
    merged["LC_ALL"] = "C"
    # A service launched from a Git hook must not write into that hook's index
    # or object store, outside the measured clone directory.
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        merged.pop(name, None)
    merged["GIT_CONFIG_COUNT"] = str(len(entries))
    for index, (key, value) in enumerate(entries):
        merged[f"GIT_CONFIG_KEY_{index}"] = key
        merged[f"GIT_CONFIG_VALUE_{index}"] = value
    return merged


async def finish_cleanup(awaitable):
    """Finish cleanup before propagating cancellation, including repeated cancels."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            # wait() neither cancels the task nor re-raises its error: that comes below.
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    try:
        result = task.result()
    except Exception as error:
        if cancelled:
            raise asyncio.CancelledError from error
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


async def stop_process(spawning, communication):
    """Kill a subprocess and its helpers, then reap it and drain the inherited pipes."""
    process = await spawning
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        try:
            # Also kill helpers when Git itself has already exited but a helper
            # still holds the pipes open. The group belongs only to this command.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The process group has already exited.
    if communication is not None:
        await communication
    else:
        await process.communicate()
    await process.wait()


async def _exec_git(args, cwd: Path | None = None, env: dict | None = None) -> tuple:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise CodeRepositoryError(
            message="git is required to clone remote repositories but was not found on PATH."
        )
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            git_binary,
            *args,
            cwd=str(cwd) if cwd else None,
            env=_git_env(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    )
    communication = None
    refused = asyncio.Event()
    refusal = None

    async def read_stderr(stream):
        chunks = []
        tail = b""
        while chunk := await stream.read(65536):
            chunks.append(chunk)
            combined = tail + chunk
            if _FILTER_REFUSED in combined:
                refused.set()
            tail = combined[-len(_FILTER_REFUSED) :]
        return b"".join(chunks)

    async def communicate(process):
        stdout, stderr = await asyncio.gather(process.stdout.read(), read_stderr(process.stderr))
        await process.wait()
        return stdout, stderr

    try:
        # Keep the spawn awaitable: cancellation while the subprocess transport
        # is being created must still leave us a process to terminate and reap.
        process = await asyncio.shield(spawning)
        communication = asyncio.create_task(communicate(process))
        refusal = asyncio.create_task(refused.wait())
        done, _pending = await asyncio.wait(
            {communication, refusal},
            timeout=_GIT_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if refused.is_set():
            raise CodeRepositoryError(
                message="The Git server ignored the partial clone filter; transfer stopped."
            )
        if not done:
            raise asyncio.TimeoutError
        stdout, stderr = await communication
    except (asyncio.CancelledError, CodeRepositoryError):
        await finish_cleanup(stop_process(spawning, communication))
        raise
    except asyncio.TimeoutError:
        await finish_cleanup(stop_process(spawning, communication))
        raise CodeRepositoryError(message=f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s.")
    finally:
        if refusal is not None:
            refusal.cancel()
            await finish_cleanup(asyncio.gather(refusal, return_exceptions=True))
    return (
        process.returncode,
        stdout.decode(errors="surrogateescape"),
        stderr.decode(errors="replace"),
    )


async def _run_git(args, cwd: Path | None = None, env: dict | None = None) -> tuple:
    returncode, _stdout, stderr = await _exec_git(args, cwd, env)
    return returncode, stderr


async def _configured_ssh_command() -> str | None:
    """The user's ``core.sshCommand`` from the global or system git config, or None.

    The GIT_SSH_COMMAND that :func:`_noninteractive_env` sets overrides that
    setting, so it is read here and carried into that command. A repository's
    own config is not read: no clone sets one, and the process's working
    directory may sit in an unrelated repository.
    """
    for scope in ("--global", "--system"):
        returncode, stdout, _stderr = await _exec_git(["config", scope, "--get", "core.sshCommand"])
        if returncode == 0 and stdout.strip():
            return stdout.strip()
    return None


async def _has_commit(clone: Path) -> bool:
    """Whether ``clone`` has a checked-out commit; an interrupted clone has none."""
    returncode, _stdout, _stderr = await _exec_git(
        ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"], cwd=clone
    )
    return returncode == 0


@asynccontextmanager
async def _clone_lock(target: Path):
    """Hold the lock of the clone directory ``target``, across processes and tasks.

    An OS file lock (``filelock``) in the clones directory's lock directory,
    released by the OS if its holder crashes. Nonblocking attempts yield between
    retries, so cancellation leaves no background thread waiting for the lock.
    Waiting is bounded by the same timeout as a Git command. The lock file stays
    in place: unlinking it could let two callers lock different inodes.
    """
    from filelock import FileLock, Timeout

    lock_dir = target.parent / _LOCKS_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_dir / f"{target.name}.lock"), thread_local=False)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _GIT_TIMEOUT_SECONDS
    while True:
        try:
            lock.acquire(blocking=False)
            break
        except Timeout:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.error("Timed out waiting for the clone lock of %s", target)
                raise CodeRepositoryError(
                    message=f"Timed out waiting for the clone lock of '{target.name}' "
                    f"after {_GIT_TIMEOUT_SECONDS}s."
                )
            await asyncio.sleep(min(0.05, remaining))
    try:
        yield
    finally:
        lock.release()


async def _claim_graph_name(base: Path, clean_url: str) -> None:
    """Refuse ``clean_url`` when another repository already owns its readable name.

    The readable slug (:func:`_readable_slug`) names the repository's code graph
    and manifest row (:func:`clone_repo_name`, :func:`repo_identity`), and two
    different repositories can share one. Indexed in one dataset they would
    delete each other's graph on every run. The first repository to use a name
    claims it by an atomically created file in the clones directory; a
    repository cloned before claims existed is recognised by the origin of its
    old clone directory.
    """
    slug = _readable_slug(clean_url)
    key = _repo_key(clean_url)
    names = base / _NAMES_DIR
    names.mkdir(parents=True, exist_ok=True)
    # Named by a hash of the exact slug: a case-insensitive file system must not
    # merge 'Org' and 'org' (different graph names), and any slug length fits.
    claim = names / hashlib.sha256(slug.encode()).hexdigest()
    if not claim.exists():
        owner = key
        legacy = base / slug
        if (legacy / ".git").is_dir():
            origin = await _origin_url(legacy)
            if origin is not None:
                owner = _repo_key(redact_repo_spec(origin))
        descriptor, pending_name = tempfile.mkstemp(prefix=".pending-", dir=names)
        pending = Path(pending_name)
        try:
            with os.fdopen(descriptor, "w") as pending_file:
                pending_file.write(owner)
            try:
                os.link(pending, claim)  # fails if another caller claimed it first
            except FileExistsError:
                pass
            except OSError as error:
                raise CodeRepositoryError(
                    message=f"Could not record the code graph name of '{clean_url}': {error}"
                ) from error
        finally:
            pending.unlink()
    owner = claim.read_text()
    if owner != key:
        raise CodeRepositoryError(
            message=f"'{clean_url}' cannot be indexed: its code graph name '{slug}' is "
            f"already used by '{owner}'."
        )


async def _origin_url(clone: Path) -> str | None:
    """The remote URL stored for ``origin`` in ``clone``, as written (no
    ``insteadOf`` rewrite), or None when it has none."""
    returncode, stdout, _stderr = await _exec_git(
        ["config", "--get", "remote.origin.url"], cwd=clone
    )
    return stdout.strip() if returncode == 0 else None


async def _remote_ref_missing(remote: str, branch: str, env: dict | None) -> bool:
    """Whether the remote answers and has no branch called ``branch``.

    ``ls-remote --exit-code`` ends with status 2 exactly when it reached the remote
    and nothing matched, which git's error text, by locale and version, cannot tell.
    """
    returncode, _stderr = await _run_git(
        ["ls-remote", "--exit-code", remote, f"refs/heads/{branch}"], env=env
    )
    return returncode == 2


async def resolve_repo_source(
    spec: str | Path,
    clones_dir: Path | None = None,
    credentials: str | None = None,
    branch: str | None = None,
    include_documents: bool = True,
) -> Path:
    """Return a local directory for the repo spec, shallow-cloning remote URLs.

    Local paths are validated and returned as-is. Remote URLs are cloned with
    ``--depth 1`` into ``clones_dir`` (default :func:`default_clones_dir`); an
    existing clone is reused only after fetching and resetting to the remote tip.
    With no ``branch``, each refresh follows the remote's current default branch.
    Remote resolution honors ``ALLOW_HTTP_REQUESTS=false``.

    ``branch`` clones that branch instead of the remote's default, into a directory of
    its own (see :func:`_clone_slug`); reusing it fetches and resets, which follows a
    force push. A branch the remote does not have raises, and a local path rejects
    ``branch``.

    ``credentials`` is the preferred way to authenticate against a private
    https remote (e.g. a GitHub App installation token): it reaches git only
    through environment-level config (see :func:`_credential_env`), so the
    URL — and everything derived from it (slug, remote, logs, errors) —
    never carries a secret. Embedding credentials in the URL userinfo still
    works but is the legacy path.

    Remote clones download trees first, then sparse-checkout selected paths at one
    commit. ``include_documents=False`` omits binary document assets; text stays
    available to enola. Disk limits are operator settings COGNEE_REPOS_MAX_BYTES
    and COGNEE_REPOS_MIN_FREE_BYTES, shared by all clones under ``clones_dir``.

    One clone serves every caller of a URL: a call with the other ``include_documents``
    mode, or a refresh the budget interrupts, changes or removes it while an earlier
    caller may still be reading it. Route one URL to one worker, in one mode at a time.
    """
    from cognee.tasks.code_graph.repo_resources import with_repository_disk_budget

    async def resolve():
        return await _resolve_repo_source(spec, clones_dir, credentials, branch, include_documents)

    if not is_remote_repo(spec):
        return await resolve()
    base = Path(clones_dir) if clones_dir else default_clones_dir()
    return await with_repository_disk_budget(resolve, base)


async def _resolve_repo_source(spec, clones_dir, credentials, branch, include_documents) -> Path:
    from cognee.tasks.code_graph.repo_resources import RepositoryResourceError

    if not is_remote_repo(spec):
        if branch is not None:
            raise CodeRepositoryError(
                message=f"branch is only supported for remote repositories, but '{spec}' is a "
                "local path. Check the branch out in that directory instead."
            )
        from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path

        # Repo specs can arrive from outside the SDK (CLI arguments, API
        # callers), so local paths take the same containment check as
        # ingestion's local-file reads (enforced only when
        # COGNEE_ALLOWED_LOCAL_FILE_ROOTS is set).
        try:
            path = resolve_local_path(spec)
        except ValueError:
            raise CodeRepositoryError(
                message=f"Repository path '{spec}' is outside the allowed local roots. "
                "Add its root to COGNEE_ALLOWED_LOCAL_FILE_ROOTS to index it."
            )
        if not path.is_dir():
            raise CodeRepositoryError(
                message=f"Repository path '{spec}' is not a directory. "
                "Pass a local repo path or a remote git URL."
            )
        return path

    if os.getenv("ALLOW_HTTP_REQUESTS", "true").strip().lower() in _FALSEY:
        raise CodeRepositoryError(
            message="Cannot clone a remote repository: outbound HTTP requests are "
            "disabled (ALLOW_HTTP_REQUESTS=false). Clone it yourself and pass the local path."
        )

    url = str(spec)

    # SSRF: the URL is caller-supplied and git runs server-side, so an http(s)
    # remote must clear the same outbound check add()'s http items already clear --
    # resolve the host and refuse internal/reserved addresses. Without it,
    # repositories=['http://169.254.169.254/...'] issues the request from inside the
    # VPC and git's stderr (returned to the caller below) distinguishes open ports,
    # live hosts and auth failures: an authenticated internal port scanner.
    #
    # ssh:// and git@host: are deliberately not routed through it: that helper only
    # understands http/https, and internal git servers over SSH are a legitimate and
    # common setup. Non-http, non-ssh transports (file://, ext::, which git would
    # execute) never reach here -- they do not match _REMOTE_PREFIXES and are handled
    # as local paths above.
    if url.lower().startswith(("http://", "https://")):
        from cognee.tasks.web_scraper.ssrf_protection import (
            SSRFProtectionError,
            validate_outbound_url,
        )

        try:
            await validate_outbound_url(url)
        except SSRFProtectionError as error:
            raise CodeRepositoryError(
                message=f"Refusing to clone '{redact_repo_spec(url)}': {error}"
            ) from error

    # Slug, remote, logs, and errors all use the credential-free URL: tokens
    # in the userinfo are short-lived, so persisting one anywhere would both
    # leak it and (via the slug) mint a new clone dir per sync.
    clean_url = redact_repo_spec(url)
    has_credentials = clean_url != url

    def _scrub(text: str) -> str:
        # git error output often echoes the URL, token included. Strip the
        # exact spec first, then any other URL userinfo git may print (e.g.
        # a redirect target or a credential-helper rewrite of the URL).
        if has_credentials:
            text = text.replace(url, clean_url)
        # git echoes the clone directory ("Cloning into '<path>'"): keep the server's
        # layout out of what the caller sees.
        for root in {str(base), os.path.realpath(base)}:
            text = text.replace(root, "<clones>")
        return re.sub(r"://[^/\s@]+@", "://", text)

    is_http_remote = clean_url.lower().startswith(("http://", "https://"))
    if credentials and not is_http_remote:
        # They would reach git as an http header, which an ssh remote never sends:
        # cloning without them would still mark the clone as credentialed.
        raise CodeRepositoryError(
            message=f"credentials are only supported for http(s) remotes, but "
            f"'{clean_url}' is an ssh remote: configure an ssh key for it instead, "
            "or pass its https:// URL."
        )

    # The caller's credentials, if any, only ever reach the remote's own host.
    git_env = _credential_env(credentials, clean_url) if credentials else {}
    ssh_overridden = "GIT_SSH_COMMAND" in os.environ or "GIT_SSH" in os.environ
    if not ssh_overridden and not is_http_remote:
        # ssh runs only for an ssh remote: the user's core.sshCommand is read only then.
        git_env = {**_noninteractive_env(await _configured_ssh_command()), **git_env}
    git_env = git_env or None
    credentialed = has_credentials or bool(credentials)

    base = Path(clones_dir) if clones_dir else default_clones_dir()
    git_env = {**(git_env or {}), "TMPDIR": str(base.resolve())}
    target = clone_dir(clean_url, branch, base)
    # The slug regex already forbids separators and dot-runs; keep an
    # explicit containment check so a clone can never land outside the
    # clones directory regardless of what the URL decomposed into.
    base_real = os.path.realpath(base)
    if not os.path.realpath(target).startswith(base_real.rstrip(os.sep) + os.sep):
        raise CodeRepositoryError(message=f"Could not derive a safe clone name for {clean_url}.")

    def _branch_not_found() -> CodeRepositoryError:
        return CodeRepositoryError(message=f"Branch '{branch}' was not found in '{clean_url}'.")

    base.mkdir(parents=True, exist_ok=True)
    # One caller at a time per clone directory, in this process or another: a
    # refresh never runs on a clone another caller is still writing or refreshing.
    async with _clone_lock(target):
        # Left by a holder that died mid-clone; nobody else uses them while we hold the lock.
        for leftover in target.parent.glob(f".{target.name}.tmp-*"):
            await finish_cleanup(asyncio.to_thread(shutil.rmtree, leftover))

        if (target / ".git").is_dir():
            # The slug is shared by every caller, so never index a clone of another
            # repository that happens to sit where this one would.
            origin = await _origin_url(target)
            if origin is None or _repo_key(redact_repo_spec(origin)) != _repo_key(clean_url):
                logger.error("The clone at %s is not a clone of %s", target, clean_url)
                raise CodeRepositoryError(
                    message=f"The existing clone of '{clean_url}' is not a clone of it "
                    f"(its origin is '{redact_repo_spec(origin) if origin else 'unset'}'). "
                    "Remove it from the clones directory to clone the repository again."
                )
            if await _has_commit(target):
                try:
                    await _refresh_clone(
                        target,
                        url,
                        clean_url,
                        branch,
                        has_credentials,
                        credentialed,
                        git_env,
                        _scrub,
                        include_documents,
                    )
                except (asyncio.CancelledError, RepositoryResourceError):
                    # A failed refresh can leave packs/index files behind. This is
                    # an SDK-owned cache; rebuild it on retry rather than accumulating output.
                    await finish_cleanup(asyncio.to_thread(shutil.rmtree, target))
                    raise
                await _claim_graph_name(base, clean_url)
                return target
            # Interrupted before it had a commit (made before clones were renamed
            # into place): never index it, clone it again instead.
            logger.warning("Replacing the incomplete clone at %s", target)
            await finish_cleanup(asyncio.to_thread(shutil.rmtree, target))

        if target.exists():
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
            else:
                logger.error("%s exists and is not a clone of %s", target, clean_url)
                raise CodeRepositoryError(
                    message=f"The clone directory of '{clean_url}' exists and is not a clone "
                    "of it. Remove it from the clones directory to clone the repository."
                )

        # Checked first: 'clone --branch' falls back to a tag of that name and leaves HEAD
        # detached, which no later fetch of the branch would follow.
        if branch is not None and await _remote_ref_missing(url, branch, git_env):
            raise _branch_not_found()
        logger.info("Cloning %s into %s", clean_url, target)
        # Cloned beside the target and renamed into place once the remote and the
        # marker are written: the target directory is always a complete clone, so
        # an interrupted clone is never found there and taken for one.
        partial = _make_partial_clone_dir(target)
        try:
            # core.symlinks=false: git otherwise materializes symlinks committed in the
            # remote, and nothing re-validates paths under the clone afterwards -- a repo
            # containing '.enola -> ~/.ssh' would have the enola snapshot written through
            # it, and 'mod.py -> /proc/self/environ' would be read into the code graph.
            # With this set, git writes each symlink as a regular file containing its
            # target path. Passed via -c on clone so it is persisted into the new repo's
            # config and later resets honour it too.
            branch_args = ["--branch", branch] if branch is not None else []
            returncode, stderr = await _run_git(
                [
                    "clone",
                    "-c",
                    "core.symlinks=false",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--no-recurse-submodules",
                    *branch_args,
                    url,
                    str(partial),
                ],
                env=git_env,
            )
            if returncode != 0:
                raise CodeRepositoryError(
                    message=f"Failed to clone '{clean_url}': {_scrub(stderr[-1000:])}"
                )
            await _checkout_snapshot(partial, "HEAD", include_documents, git_env, _scrub)
            if has_credentials:
                returncode, stderr = await _run_git(
                    ["remote", "set-url", "origin", clean_url], cwd=partial
                )
                if returncode != 0:
                    # The stored remote would keep the token: never move this clone in.
                    raise CodeRepositoryError(
                        message=f"Could not store the remote of '{clean_url}': "
                        f"{_scrub(stderr[-500:])}"
                    )
            if credentialed and (partial / ".git").is_dir():
                (partial / ".git" / _CREDENTIALED_MARKER).touch()
            # Claimed once the clone succeeded: a URL that does not clone never takes a name.
            await _claim_graph_name(base, clean_url)
            os.replace(partial, target)
        finally:
            if partial.exists():
                await finish_cleanup(asyncio.to_thread(shutil.rmtree, partial))
        return target


def _make_partial_clone_dir(target: Path) -> Path:
    """A new empty directory beside ``target`` that a clone is made in before its rename."""
    partial = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    # mkdtemp makes it 0700 and the rename would carry that onto the clone, where
    # git clone itself leaves the mode of a new directory: match the clones directory.
    partial.chmod(stat.S_IMODE(target.parent.stat().st_mode))
    return partial


async def _refresh_clone(
    target: Path,
    url: str,
    clean_url: str,
    branch: str | None,
    has_credentials: bool,
    credentialed: bool,
    git_env: dict | None,
    scrub,
    include_documents: bool,
) -> None:
    """Bring the existing clone ``target`` up to date; never return stale success."""
    marker = target / ".git" / _CREDENTIALED_MARKER
    # Legacy URL-embedded credentials: fetch from the explicit URL
    # instead of the stored remote (which is deliberately
    # credential-free). Out-of-band credentials ride the environment.
    # Fetch the remote HEAD rather than the clone's old upstream: the default
    # branch may have changed. Explicit branches need a full ref to avoid tags
    # and refspec syntax. Reset also handles force-pushed history.
    remote = url if has_credentials else "origin"
    ref = f"refs/heads/{branch}" if branch is not None else "HEAD"
    returncode, stderr = await _run_git(
        ["fetch", "--depth", "1", "--filter=blob:none", "--no-recurse-submodules", remote, ref],
        cwd=target,
        env=git_env,
    )
    if returncode != 0 and branch is not None and await _remote_ref_missing(url, branch, git_env):
        raise CodeRepositoryError(message=f"Branch '{branch}' was not found in '{clean_url}'.")
    if returncode == 0:
        checkout_env = dict(git_env or {})
        if has_credentials:
            # Lazy blob fetches use origin, even though the initial fetch used the
            # token URL. Override it only for this process, never in stored config.
            index = len(_git_config_entries(checkout_env))
            checkout_env.update(
                {
                    "GIT_CONFIG_COUNT": str(index + 1),
                    f"GIT_CONFIG_KEY_{index}": "remote.origin.url",
                    f"GIT_CONFIG_VALUE_{index}": url,
                }
            )
        await _checkout_snapshot(target, "FETCH_HEAD", include_documents, checkout_env, scrub)
    if returncode != 0:
        # A clone made with credentials may be private: it is handed out only
        # through a refresh that proves access. Credentials the caller passed
        # prove nothing until the remote accepts them -- they may be wrong.
        if marker.is_file():
            raise CodeRepositoryError(
                message=f"Could not refresh '{clean_url}', and its existing clone was made "
                "with credentials, so it is used only after a refresh with valid credentials "
                f"succeeds: {scrub(stderr[-500:])}"
            )
        raise CodeRepositoryError(
            message=f"Could not refresh '{clean_url}': {scrub(stderr[-500:])}"
        )
    elif credentialed:
        marker.touch()


async def _checkout_snapshot(
    repo: Path, ref: str, include_documents: bool, env: dict, scrub
) -> None:
    from cognee.tasks.code_graph.repo_resources import check_disk, repository_disk_config
    from cognee.tasks.code_graph.repo_selection import (
        check_selected_content,
        select_tree,
        write_repository_snapshot,
        write_sparse_patterns,
    )

    code, commit, stderr = await _exec_git(
        ["rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo, env=env
    )
    commit = commit.strip()
    if code or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise CodeRepositoryError(
            message=f"Could not resolve repository commit: {scrub(stderr[-500:])}"
        )
    code, tree, stderr = await _exec_git(
        ["ls-tree", "-rz", "--full-tree", commit], cwd=repo, env=env
    )
    if code:
        raise CodeRepositoryError(message=f"Could not list repository tree: {scrub(stderr[-500:])}")
    paths, skipped = select_tree(tree, include_documents)
    await asyncio.to_thread(write_sparse_patterns, repo, paths)
    for args in (
        ["config", "remote.origin.promisor", "true"],
        ["config", "remote.origin.partialclonefilter", "blob:none"],
        ["config", "core.sparseCheckout", "true"],
        ["config", "core.sparseCheckoutCone", "false"],
        ["reset", "--hard", commit],
    ):
        code, stderr = await _run_git(args, cwd=repo, env=env)
        if code:
            raise CodeRepositoryError(
                message=f"Could not checkout repository: {scrub(stderr[-500:])}"
            )
    await asyncio.to_thread(check_selected_content, repo, paths)
    snapshot = {
        "commit": commit,
        "include_documents": include_documents,
        "selected_files": len(paths),
        "skipped": skipped,
    }
    write_repository_snapshot(repo, snapshot)
    await asyncio.to_thread(check_disk, repo.parent, repository_disk_config())
    logger.info("Repository %s: selected %d files; skipped %s", commit, len(paths), skipped)
