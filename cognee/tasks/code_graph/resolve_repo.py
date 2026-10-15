"""Resolve repository specs (local paths or remote git URLs) to local clones.

Used by ``add()`` to recognise a GitHub/GitLab repository URL
(:func:`code_repo_clone_url`) and shallow-clone it for the enola code-graph
route, and by the GitHub integration sync to clone private repositories with
an installation token (``resolve_repo_source(credentials=...)``). Clones live under
``BaseConfig.repos_root_directory`` (``~/.cognee/repos`` by default) and are
reused across calls once its ``origin`` is checked; an existing clone is
refreshed with a best-effort ``git pull`` (a clone made with credentials is
refused when that refresh fails). A ``branch`` can be chosen for a remote
repository; each branch is cloned into a directory of its own. A new clone is
made in a temporary sibling directory and renamed into place once complete, and
all work on one clone directory is serialized by a file lock, across processes.
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import os
import re
import shutil
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
# The ssh half of the above, as one whitespace-free token: unambiguously a git
# remote, never a web page -- and never a sentence that merely starts with one.
_SSH_REPO_SPEC = re.compile(r"(?:git@[\w.-]+:|ssh://)\S+")

# Written into .git of a clone made with credentials. Such a clone may hold a
# private repository, so it is handed out only after a refresh that proves the
# caller's access: a failed refresh is refused instead of handing over the stale
# copy, whatever credentials the caller passed (they may be wrong).
_CREDENTIALED_MARKER = "cognee-credentialed"

# Per-clone lock files, in a directory of their own inside the clones directory.
_LOCKS_DIR = ".locks"

_FALSEY = {"false", "0", "no", "off"}

_GIT_TIMEOUT_SECONDS = 600

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


def is_ssh_repo_spec(spec) -> bool:
    """Whether the spec is an ssh-style git remote (``git@host:owner/repo``, ``ssh://``).

    These name a repository and nothing else -- unlike an http(s) URL, which may
    be a web page -- but :func:`code_repo_clone_url` does not accept them, so
    ingestion would otherwise store the spec string as a text document. Callers
    refuse them with the instruction in :data:`SSH_REPO_SPEC_MESSAGE`.
    """
    return isinstance(spec, str) and _SSH_REPO_SPEC.fullmatch(spec.strip()) is not None


SSH_REPO_SPEC_MESSAGE = (
    "ssh git remotes (git@host:owner/repo, ssh://...) are not recognised as repositories "
    'by add(): pass codegraph_config={"treat_as_repository": True} to clone this one as a '
    "repository, or give its https:// URL or a local clone's directory instead."
)


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


async def _exec_git(args, cwd: Path | None = None, env: dict | None = None) -> tuple:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise CodeRepositoryError(
            message="git is required to clone remote repositories but was not found on PATH."
        )
    process = await asyncio.create_subprocess_exec(
        git_binary,
        *args,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, **_noninteractive_env(), **(env or {})},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_GIT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise CodeRepositoryError(message=f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s.")
    return process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


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
    released by the OS if its holder crashes. Acquire and release run on one
    dedicated thread, off the event loop, as in ``cognee.modules.migrations.runner``
    and for the same reason: filelock's per-thread state breaks when they run on
    different threads. A caller cancelled while waiting releases the lock as soon
    as its thread gets it.
    """
    from filelock import FileLock

    lock_dir = target.parent / _LOCKS_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_dir / f"{target.name}.lock"), thread_local=False)
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cognee-clone-lock"
    )
    acquiring = loop.run_in_executor(executor, lock.acquire)
    try:
        await asyncio.shield(acquiring)
    except asyncio.CancelledError:

        def _release_late(future) -> None:
            if not future.cancelled() and future.exception() is None:
                executor.submit(lock.release)
            executor.shutdown(wait=False)

        acquiring.add_done_callback(_release_late)
        raise
    except BaseException:
        executor.shutdown(wait=False)
        raise
    try:
        yield
    finally:
        try:
            await loop.run_in_executor(executor, lock.release)
        finally:
            executor.shutdown(wait=False)


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
) -> Path:
    """Return a local directory for the repo spec, shallow-cloning remote URLs.

    Local paths are validated and returned as-is. Remote URLs are cloned with
    ``--depth 1`` into ``clones_dir`` (default :func:`default_clones_dir`); an
    existing clone is reused after a best-effort ``git pull --ff-only``.
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
    """
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
        return re.sub(r"://[^/\s@]+@", "://", text)

    # The caller's credentials, if any, only ever reach the remote's own host.
    git_env = _credential_env(credentials, clean_url) if credentials else {}
    ssh_overridden = "GIT_SSH_COMMAND" in os.environ or "GIT_SSH" in os.environ
    if not ssh_overridden and not clean_url.lower().startswith(("http://", "https://")):
        # ssh runs only for an ssh remote: the user's core.sshCommand is read only then.
        git_env = {**_noninteractive_env(await _configured_ssh_command()), **git_env}
    git_env = git_env or None
    credentialed = has_credentials or bool(credentials)

    base = Path(clones_dir) if clones_dir else default_clones_dir()
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
            await asyncio.to_thread(shutil.rmtree, leftover, True)

        if (target / ".git").is_dir():
            # The slug is shared by every caller, so never index a clone of another
            # repository that happens to sit where this one would.
            origin = await _origin_url(target)
            if origin is None or _repo_key(redact_repo_spec(origin)) != _repo_key(clean_url):
                raise CodeRepositoryError(
                    message=f"The existing clone at {target} is not a clone of '{clean_url}' "
                    f"(its origin is '{redact_repo_spec(origin) if origin else 'unset'}'). "
                    "Remove that directory to clone the repository again."
                )
            if await _has_commit(target):
                await _refresh_clone(
                    target, url, clean_url, branch, has_credentials, credentialed, git_env, _scrub
                )
                return target
            # Interrupted before it had a commit (made before clones were renamed
            # into place): never index it, clone it again instead.
            logger.warning("Replacing the incomplete clone at %s", target)
            await asyncio.to_thread(shutil.rmtree, target)

        if target.exists():
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
            else:
                raise CodeRepositoryError(
                    message=f"{target} exists and is not a clone of '{clean_url}'. "
                    "Remove it to clone the repository."
                )

        # Checked first: 'clone --branch' falls back to a tag of that name and leaves HEAD
        # detached, which no later fetch of the branch would follow.
        if branch is not None and await _remote_ref_missing(url, branch, git_env):
            raise _branch_not_found()
        logger.info("Cloning %s into %s", clean_url, target)
        # Cloned beside the target and renamed into place once the remote and the
        # marker are written: the target directory is always a complete clone, so
        # an interrupted clone is never found there and taken for one.
        partial = Path(
            await asyncio.to_thread(
                tempfile.mkdtemp, prefix=f".{target.name}.tmp-", dir=target.parent
            )
        )
        try:
            # core.symlinks=false: git otherwise materializes symlinks committed in the
            # remote, and nothing re-validates paths under the clone afterwards -- a repo
            # containing '.enola -> ~/.ssh' would have the enola snapshot written through
            # it, and 'mod.py -> /proc/self/environ' would be read into the code graph.
            # With this set, git writes each symlink as a regular file containing its
            # target path. Passed via -c on clone so it is persisted into the new repo's
            # config and the later 'git pull --ff-only' honours it too.
            branch_args = ["--branch", branch] if branch is not None else []
            returncode, stderr = await _run_git(
                [
                    "clone",
                    "-c",
                    "core.symlinks=false",
                    "--depth",
                    "1",
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
            os.replace(partial, target)
        finally:
            if partial.exists():
                await asyncio.to_thread(shutil.rmtree, partial, True)
        return target


async def _refresh_clone(
    target: Path,
    url: str,
    clean_url: str,
    branch: str | None,
    has_credentials: bool,
    credentialed: bool,
    git_env: dict | None,
    scrub,
) -> None:
    """Bring the existing clone ``target`` up to date, or decide it may be used stale."""
    marker = target / ".git" / _CREDENTIALED_MARKER
    # Legacy URL-embedded credentials: fetch from the explicit URL
    # instead of the stored remote (which is deliberately
    # credential-free). Out-of-band credentials ride the environment.
    if branch is None:
        pull_args = ["pull", "--ff-only"] + ([url] if has_credentials else [])
        returncode, stderr = await _run_git(pull_args, cwd=target, env=git_env)
    else:
        # Fetch and reset, not pull: a rebased or force-pushed branch is not a
        # fast-forward. The ref is spelled out because git resolves a bare name
        # itself (a tag before a branch, '+x' as a force refspec).
        remote = url if has_credentials else "origin"
        fetch_args = ["fetch", "--depth", "1", remote, f"refs/heads/{branch}"]
        returncode, stderr = await _run_git(fetch_args, cwd=target, env=git_env)
        if returncode != 0 and await _remote_ref_missing(url, branch, git_env):
            raise CodeRepositoryError(message=f"Branch '{branch}' was not found in '{clean_url}'.")
        if returncode == 0:
            returncode, stderr = await _run_git(["reset", "--hard", "FETCH_HEAD"], cwd=target)
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
        # A stale clone is still indexable; the caller asked for the repo,
        # not for freshness guarantees.
        logger.warning(
            "Reusing existing clone at %s (refresh failed: %s)", target, scrub(stderr[-500:])
        )
    elif credentialed:
        marker.touch()
