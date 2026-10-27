"""Resolve remote repositories for add() into verified, locked local clones.

Refresh failures raise rather than returning cached private content. Readers must
hold the same clone lock until they finish consuming the checkout.
"""

import asyncio
import base64
import hashlib
import os
import re
import shutil
import stat
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
from weakref import WeakKeyDictionary

from fastapi import status

from cognee.base_config import get_base_config
from cognee.exceptions import CogneeSystemError
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.process import ProcessOutputError, finish_cleanup, run_process

logger = get_logger("code_graph")

# How often a caller waiting for a clone lock says so.
_CLONE_LOCK_LOG_SECONDS = 30.0

_REMOTE_PREFIXES = ("https://", "http://", "git@", "ssh://")

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
    or logged, including clone slugs, stored git remotes and errors, must use this form instead of the raw spec.
    """
    url = str(spec)
    if "://" not in url:
        return url
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def repo_key(url: str) -> str:
    """Canonical host and full repository path, without credentials or transport.

    GitHub and GitLab paths are case-insensitive. Other hosts retain path case.
    Ports remain part of the host so separate Git servers never share identities.
    """
    if not isinstance(url, str) or not is_remote_repo(url):
        raise ValueError("Expected a remote repository URL.")
    if url.startswith("git@"):
        host, separator, path = url[4:].partition(":")
        if not separator:
            raise ValueError("Expected an SSH repository path.")
        parts = urlsplit(f"ssh://{host}/{path}")
    else:
        parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path.strip("/").removesuffix(".git")
    if not host or not path:
        raise ValueError("Expected a repository host and path.")
    host = host.removeprefix("www.")
    if host in {"github.com", "gitlab.com"}:
        path = path.lower()
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return f"{host}/{path}"


def repo_identity(url: str) -> str:
    """Full repository-key hash, independent of branch and clone directory."""
    return hashlib.sha256(repo_key(url).encode()).hexdigest()


def _readable_slug(url: str) -> str:
    """The old readable clone name, for recognizing legacy repository records."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", repo_key(url)).strip("-.")


def _clone_slug(url: str, branch: str | None = None) -> str:
    """A filesystem-safe digest of the repository and selected branch."""
    if branch is not None:
        check_branch_name(branch)
    key = repo_key(url)
    return hashlib.sha256(f"{key}\0{branch if branch is not None else ''}".encode()).hexdigest()


def clone_dir(url: str, branch: str | None = None, clones_dir: Path | None = None) -> Path:
    """Managed checkout location for a remote repository and branch."""
    return (Path(clones_dir) if clones_dir is not None else default_clones_dir()) / _clone_slug(
        url, branch
    )


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


def _credential_env(token: str | None, url: str, username: str = "x-access-token") -> dict:
    """Reset saved HTTP headers and supply optional credentials for this repo path.

    Secrets travel through environment configuration, never argv or .git/config.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        return {}
    host = parts.netloc.rsplit("@", 1)[-1].lower()
    key = f"http.{scheme}://{host}{parts.path.rstrip('/')}/.extraHeader"
    authorization = "Authorization:"
    if token is not None:
        basic = base64.b64encode(f"{username}:{token}".encode()).decode()
        authorization = f"Authorization: Basic {basic}"
    config = {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": key,
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": key,
        "GIT_CONFIG_VALUE_1": authorization,
    }
    return config


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
# program. Without these a
# refresh of a private clone could succeed on the server's stored credentials (a
# helper, a keychain) for a caller who brought none, and unlock that clone. An
# empty credential.helper resets the helper list built from the system, global
# and repository config, which git reads before the environment's entries.
_NO_STORED_CREDENTIALS_CONFIG = (("credential.helper", ""), ("core.askPass", ""))

# A fetch can leave a detached ``maintenance`` or ``gc`` process behind that outlives the
# clone lock and holds git lock files of its own.
_NO_BACKGROUND_MAINTENANCE_CONFIG = (("maintenance.auto", "false"), ("gc.auto", "0"))


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
    no stored credentials (see :data:`_NO_STORED_CREDENTIALS_CONFIG`) and no background
    maintenance (see :data:`_NO_BACKGROUND_MAINTENANCE_CONFIG`).

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
        + [("filter.lfs.process", ""), ("filter.lfs.smudge", ""), ("filter.lfs.required", "false")]
        # Git for Windows must support long pack/promisor paths in hashed clone directories.
        + [("core.longpaths", "true")]
        + list(_NO_BACKGROUND_MAINTENANCE_CONFIG)
    )
    merged = {
        name: value
        for source in (os.environ, _noninteractive_env(), extra)
        for name, value in source.items()
        if not _is_git_config_entry(name)
    }
    merged["GIT_ASKPASS"] = ""
    merged["SSH_ASKPASS"] = ""
    merged["GIT_LFS_SKIP_SMUDGE"] = "1"
    merged["LC_ALL"] = "C"
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


def _remove_clone(path: Path) -> None:
    """Remove an owned clone, including Git's read-only objects on Windows."""

    def remove_readonly(function, filename, error_info):
        error = error_info[1]
        if os.name != "nt" or not isinstance(error, PermissionError):
            raise error
        os.chmod(filename, stat.S_IREAD | stat.S_IWRITE)
        function(filename)

    shutil.rmtree(path, onerror=remove_readonly)


def _publish_clone(partial: Path, target: Path, replace_full_clone: bool) -> None:
    """Retire an old cache and publish its validated replacement without yielding."""
    if replace_full_clone:
        _remove_clone(target)
    os.replace(partial, target)


async def _exec_git(args, cwd: Path | None = None, env: dict | None = None) -> tuple:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise CodeRepositoryError(
            message="git is required to clone remote repositories but was not found on PATH."
        )
    try:
        returncode, stdout, stderr = await run_process(
            [git_binary, *args],
            cwd=cwd,
            env=_git_env(env),
            timeout=_GIT_TIMEOUT_SECONDS,
            reject_stderr=b"filtering not recognized by server",
        )
    except ProcessOutputError as error:
        raise CodeRepositoryError(message=str(error)) from error
    except asyncio.TimeoutError:
        raise CodeRepositoryError(message=f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s.")
    if returncode and any(
        message in stderr for message in (b"No space left on device", b"Disk quota exceeded")
    ):
        from cognee.tasks.code_graph.repo_resources import RepositoryResourceError

        raise RepositoryResourceError("filesystem space or quota exhausted")
    return returncode, stdout.decode(errors="surrogateescape"), stderr.decode(errors="replace")


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


async def repository_commit(clone: Path) -> str:
    """Return the checkout's full commit ID; reject incomplete repositories."""
    code, stdout, _stderr = await _exec_git(
        ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"], cwd=clone
    )
    commit = stdout.strip()
    if code != 0 or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise CodeRepositoryError(message="The repository has no valid checked-out commit.")
    return commit


_HELD_CLONE_LOCKS: WeakKeyDictionary = WeakKeyDictionary()


@asynccontextmanager
async def _clone_lock(target: Path, *, clones_dir: Path | None = None):
    """Serialize clone writers and readers across processes, reentrant in one task.

    Child tasks must acquire the file lock themselves. POSIX lock files remain
    in place so another caller cannot lock a replacement inode; on Windows the
    filelock backend manages deletion when releasing its lock.

    ``target`` must be a clone directly inside the clones root (``clones_dir``, else
    ``COGNEE_REPOS_DIR`` read now, so moving that directory invalidates manifests that
    point into the old one) and not a symlink; this is the only place that checks it.
    """
    from filelock import FileLock, Timeout

    root_path = os.path.realpath(clones_dir if clones_dir is not None else default_clones_dir())
    root = Path(root_path)
    # Cached clones are direct children of the storage root. Reject escapes and symlink
    # aliases before checking or creating lock files.
    resolved = os.path.realpath(target)
    if not (resolved == root_path or resolved.startswith(root_path.rstrip(os.sep) + os.sep)) or (
        resolved != root_path and resolved != os.path.join(root_path, os.path.basename(target))
    ):
        raise CodeRepositoryError(message="The repository clone path is outside its storage root.")
    target = Path(resolved)
    task = asyncio.current_task()
    held = _HELD_CLONE_LOCKS.get(task)
    if held is not None and target in held:
        yield
        return
    lock_root = Path(os.path.realpath(root / ".locks"))
    if lock_root == root or not lock_root.is_relative_to(root):
        raise CodeRepositoryError(
            message="The repository lock directory is outside its storage root."
        )
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(os.path.realpath(lock_root / f"{target.name}.lock"))
    if lock_path.parent != lock_root:
        raise CodeRepositoryError(message="The repository lock file is outside its lock directory.")
    lock = FileLock(str(lock_path), thread_local=False)
    held = _HELD_CLONE_LOCKS.setdefault(task, set())
    try:
        waited_since = time.monotonic()
        while True:
            try:
                lock.acquire(blocking=False)
                break
            except Timeout:
                await asyncio.sleep(0.05)
                if time.monotonic() - waited_since >= _CLONE_LOCK_LOG_SECONDS:
                    waited_since = time.monotonic()
                    logger.warning(
                        "Still waiting for a repository clone lock: another run is using it."
                    )
        held.add(target)
        try:
            yield
        finally:
            held.remove(target)
            lock.release()
    finally:
        if not held:
            _HELD_CLONE_LOCKS.pop(task, None)


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

    Remote checkouts use blob filtering and exact sparse paths. Documents are
    included by default; code-only mode omits binary document assets while
    retaining text and configuration needed by custom extractors. Selected Git
    LFS files and servers that ignore filtering raise CodeRepositoryError.
    Operator disk guards use COGNEE_REPOS_MAX_BYTES and COGNEE_REPOS_MIN_FREE_BYTES;
    neither has a default limit. Polling requires filesystem quota headroom.

    Local paths are validated and returned as-is. Remote URLs are cloned with
    ``--depth 1`` into ``clones_dir`` (default :func:`default_clones_dir`); an
    existing clone is returned only after fetching and resetting to the remote tip.
    Remote resolution honors ``ALLOW_HTTP_REQUESTS=false``.

    ``branch`` clones that branch instead of the remote's default, into a directory of
    its own (see :func:`_clone_slug`); reusing it fetches and resets, which follows a
    force push. A branch the remote does not have raises, and a local path rejects
    ``branch``.

    ``credentials`` is the preferred way to authenticate against a private
    https remote (e.g. a GitHub App installation token): it reaches git only
    through environment-level config (see :func:`_credential_env`), so the
    URL and everything derived from it
    never carries a secret. Embedding credentials in the URL userinfo still
    works but is the legacy path.
    """
    from cognee.tasks.code_graph.repo_resources import repository_disk_budget

    if not is_remote_repo(spec):
        return await _resolve_repo_source(spec, clones_dir, credentials, branch, include_documents)
    base = Path(clones_dir) if clones_dir is not None else default_clones_dir()
    async with repository_disk_budget(base):
        return await _resolve_repo_source(spec, clones_dir, credentials, branch, include_documents)


async def _resolve_repo_source(spec, clones_dir, credentials, branch, include_documents):
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

    if branch is not None:
        check_branch_name(branch)
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
            await validate_outbound_url(redact_repo_spec(url))
        except SSRFProtectionError as error:
            raise CodeRepositoryError(
                message=f"Refusing to clone '{redact_repo_spec(url)}': {error}"
            ) from error

    clean_url = redact_repo_spec(url)
    is_http_remote = clean_url.startswith(("http://", "https://"))
    if credentials is not None and (not isinstance(credentials, str) or not credentials):
        raise ValueError("credentials must be a non-empty string.")
    if credentials and not is_http_remote:
        raise CodeRepositoryError(
            message="credentials require an http(s) repository URL.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if url.startswith("ssh://"):
        ssh_parts = urlsplit(url)
        if ssh_parts.password is not None:
            raise CodeRepositoryError(
                message="SSH repository URLs require key authentication.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        # SSH user names select the remote account and are not HTTP credentials.
        clean_url = url
    parts = urlsplit(url) if is_http_remote else None
    username = "x-access-token"
    token = credentials
    if parts is not None and parts.username is not None and credentials is None:
        username = unquote(parts.username)
        token = unquote(parts.password) if parts.password is not None else ""
    git_env = _credential_env(token, clean_url, username) if is_http_remote else {}
    if not is_http_remote and "GIT_SSH_COMMAND" not in os.environ and "GIT_SSH" not in os.environ:
        git_env.update(_noninteractive_env(await _configured_ssh_command()))
    base = Path(clones_dir) if clones_dir is not None else default_clones_dir()
    target = clone_dir(clean_url, branch, base)

    def scrub(text: str) -> str:
        text = text.replace(url, clean_url)
        if token is not None:
            encoded = base64.b64encode(f"{username}:{token}".encode()).decode()
            text = text.replace(encoded, "<redacted>")
            if token:
                text = text.replace(token, "<redacted>")
        for root in {str(base), str(base.resolve())}:
            text = text.replace(root, "<clones>")
        return re.sub(r"://[^/\s@]+@", "://", text)

    base.mkdir(parents=True, exist_ok=True)
    git_env.update(
        {"TMPDIR": str(base.resolve()), "TMP": str(base.resolve()), "TEMP": str(base.resolve())}
    )
    async with _clone_lock(target, clones_dir=base):
        for leftover in target.parent.glob(f".{target.name}.tmp-*"):
            await finish_cleanup(asyncio.to_thread(_remove_clone, leftover))
        replace_full_clone = False
        if (target / ".git").is_dir():
            await _verify_origin(target, clean_url)
            from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

            replace_full_clone = read_repository_snapshot(target) is None
            if await _has_commit(target) and not replace_full_clone:
                await asyncio.to_thread(_remove_stale_git_locks, target)
                try:
                    ref = f"refs/heads/{branch}" if branch is not None else "HEAD"
                    code, stderr = await _run_git(
                        [
                            "fetch",
                            "--depth",
                            "1",
                            "--filter=blob:none",
                            "--no-recurse-submodules",
                            clean_url,
                            ref,
                        ],
                        cwd=target,
                        env=git_env,
                    )
                    if code != 0:
                        if branch is not None and await _remote_ref_missing(
                            clean_url, branch, git_env
                        ):
                            raise CodeRepositoryError(
                                message=f"Branch '{branch}' was not found in '{clean_url}'.",
                                status_code=status.HTTP_400_BAD_REQUEST,
                            )
                        raise CodeRepositoryError(
                            message=f"Could not refresh '{clean_url}': {scrub(stderr)}"
                        )
                    # Invalidate the old record before touching the checkout. A failed
                    # refresh must never be accepted by an earlier manifest.
                    (target / ".git" / "cognee-snapshot.json").unlink(missing_ok=True)
                    await _checkout_snapshot(
                        target, "FETCH_HEAD", include_documents, git_env, scrub
                    )
                    return target
                except (asyncio.CancelledError, RepositoryResourceError):
                    await finish_cleanup(asyncio.to_thread(_remove_clone, target))
                    raise
            if not replace_full_clone:
                await finish_cleanup(asyncio.to_thread(_remove_clone, target))
        if target.exists() and not replace_full_clone:
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
            else:
                raise CodeRepositoryError(message="The repository clone path is already occupied.")
        if branch is not None and await _remote_ref_missing(clean_url, branch, git_env):
            raise CodeRepositoryError(
                message=f"Branch '{branch}' was not found in '{clean_url}'.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        partial = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            branch_args = ["--branch", branch] if branch is not None else []
            code, stderr = await _run_git(
                [
                    "clone",
                    "-c",
                    "core.symlinks=false",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--no-local",
                    "--no-recurse-submodules",
                    *branch_args,
                    clean_url,
                    str(partial),
                ],
                env=git_env,
            )
            if code != 0:
                raise CodeRepositoryError(message=f"Failed to clone '{clean_url}': {scrub(stderr)}")
            await _verify_origin(partial, clean_url)
            await _checkout_snapshot(partial, "HEAD", include_documents, git_env, scrub)
            if not await _has_commit(partial):
                raise CodeRepositoryError(
                    message="The cloned repository has no checked-out commit."
                )
            # Once retirement starts, cancellation must finish publication before
            # cleanup can delete the replacement or release the clone lock.
            await finish_cleanup(
                asyncio.to_thread(_publish_clone, partial, target, replace_full_clone)
            )
        finally:
            if partial.exists():
                await finish_cleanup(asyncio.to_thread(_remove_clone, partial))
        return target


def _remove_stale_git_locks(clone: Path) -> None:
    """Delete the lock files a killed git process left in ``clone``.

    git is killed outright on cancel and timeout (as is the whole process on an OOM kill),
    so it never removes its own ``index.lock`` or ``shallow.lock``, and every later
    fetch or reset would fail on them. The clone lock is held and git runs without
    background maintenance, so no git process of ours is running in this clone and every
    lock file here is stale. Git run by hand in a managed clone is not supported.
    """
    git_dir = clone / ".git"
    for lock in [*git_dir.glob("*.lock"), *(git_dir / "refs").rglob("*.lock")]:
        lock.unlink(missing_ok=True)


async def _verify_origin(clone: Path, expected: str) -> None:
    origin = await _origin_url(clone)
    if not origin or not is_remote_repo(origin) or repo_key(origin) != repo_key(expected):
        raise CodeRepositoryError(
            message="The existing clone does not match the requested repository. "
            "Remove the managed clone and retry."
        )


async def _checkout_snapshot(
    repo: Path, ref: str, include_documents: bool, env: dict, scrub
) -> None:
    from cognee.tasks.code_graph.repo_resources import (
        check_repository_disk,
        repository_disk_config,
        repository_storage_roots,
    )
    from cognee.tasks.code_graph.repo_selection import (
        check_selected_content,
        select_tree,
        selection_fingerprint,
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
    # A cancelled to_thread await does not stop its writer. Keep the lock until
    # it finishes so it cannot recreate a directory after clone cleanup.
    await finish_cleanup(asyncio.to_thread(write_sparse_patterns, repo, paths))
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
        "selection_fingerprint": selection_fingerprint(paths, include_documents),
        "selection_version": 1,
        "include_documents": include_documents,
        "selected_files": len(paths),
        "skipped": skipped,
    }
    await asyncio.to_thread(
        check_repository_disk, repository_storage_roots(repo.parent), repository_disk_config()
    )
    write_repository_snapshot(repo, snapshot)
    logger.info("Repository %s: selected %d files; skipped %s", commit, len(paths), skipped)
