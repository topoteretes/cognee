"""Resolve repository specs (local paths or remote git URLs) to local clones.

Used by ``remember(..., content_type="code")`` so callers can pass a GitHub
URL (or a list of them) and get the enola code-graph pipeline run on a local
shallow clone, and by ``add()`` to recognise a GitHub/GitLab repository URL
(:func:`code_repo_clone_url`) and clone it the same way. Clones live under
``BaseConfig.repos_root_directory`` (``~/.cognee/repos`` by default) and are
reused across calls; an existing clone is refreshed with a best-effort
``git pull``. A ``branch`` can be chosen for a remote repository; the clone
directory is still keyed on the URL alone (see :func:`resolve_repo_source`).
"""

import asyncio
import base64
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import status

from cognee.base_config import get_base_config
from cognee.exceptions import CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("code_graph")

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
    and ``ssh://`` specs are not detected: they have no web-page reading to
    disambiguate from and stay explicit via ``remember(content_type="code")``.

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


def _clone_slug(url: str) -> str:
    """A stable directory name for a remote URL, e.g. 'github.com-org-repo'.

    Deliberately blind to the branch: the code_repo data row is keyed on this
    directory's path, and the code graph keys node ids and the stale sweep on the
    repository's name (enola takes it from the clone's origin remote, or from this
    directory's name when there is none), so one URL stays one repository, with
    one data row, whatever branch it holds.
    """
    tail = url.split("://")[-1].replace(":", "/").rstrip("/")
    tail = tail.removesuffix(".git")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.")


# git's refname rules that cannot appear anywhere in a branch name: control
# characters, space, ~ ^ : ? * [ and backslash.
_BRANCH_FORBIDDEN_CHARS = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")

# Stands for a clone that is on no branch (a tag, a detached HEAD): it contains a space,
# so it can never equal a requested branch name.
_NOT_A_BRANCH = "(not a branch)"


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


def _symbolic_ref(git_dir: Path, ref: str) -> str | None:
    """Where the symbolic ref ``ref`` of the repository at ``git_dir`` points, or None."""
    result = subprocess.run(
        [_git_binary(), f"--git-dir={git_dir}", "symbolic-ref", "-q", ref],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_clone_branch(target: Path) -> str | None:
    """The branch a clone is on, None for a clone of the remote's default branch.

    A clone without ``--branch`` records origin's HEAD, one with ``--branch`` does not,
    and HEAD names the branch (checked on git 2.50.1); that also holds for every clone
    made before ``branch`` existed. Asked of git: with reftable, .git holds no ref files.
    """
    git_dir = target / ".git"
    if _symbolic_ref(git_dir, "refs/remotes/origin/HEAD") is not None:
        return None
    head = _symbolic_ref(git_dir, "HEAD") or ""
    prefix = "refs/heads/"
    return head.removeprefix(prefix) if head.startswith(prefix) else _NOT_A_BRANCH


def _clear_read_only_and_retry(function, path, _error) -> None:
    """``shutil.rmtree`` error handler: git writes its object files read-only, and
    Windows will not delete those, so make the path writable and try again."""
    os.chmod(path, stat.S_IRWXU)
    function(path)


def _remove_tree(path: Path) -> None:
    """Remove a directory tree, a clone's read-only files included; best effort."""
    if not os.path.lexists(path):
        return
    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_clear_read_only_and_retry)
        else:
            shutil.rmtree(path, onerror=_clear_read_only_and_retry)
    except OSError as error:
        logger.warning("Could not remove %s: %s", path, error)


# One lock per clone directory, held while it is cloned, replaced or refreshed. It is
# process-local, like the dataset lock, and waits on nothing else, so taking it under
# the dataset lock (the order add() reaches it in) cannot deadlock.
_clone_locks: dict[str, asyncio.Lock] = {}


def _clone_lock(target: Path) -> asyncio.Lock:
    return _clone_locks.setdefault(os.path.realpath(target), asyncio.Lock())


def _credential_env(token: str) -> dict:
    """Auth for git over https, injected as environment-level git config.

    The environment channel (git >= 2.31) keeps the token out of argv (which
    any local process can read from the process listing), out of every
    URL-shaped string (so nothing derived from the URL can leak it into
    logs or error messages), and out of the on-disk remote config.
    """
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
    }


def _git_binary() -> str:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise CodeRepositoryError(
            message="git is required to clone remote repositories but was not found on PATH."
        )
    return git_binary


async def _run_git(args, cwd: Path | None = None, env: dict | None = None) -> tuple:
    process = await asyncio.create_subprocess_exec(
        _git_binary(),
        *args,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, **env} if env else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_GIT_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise CodeRepositoryError(message=f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s.")
    return process.returncode, stderr.decode(errors="replace")


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

    ``branch`` clones that branch instead of the remote's default. A URL has one clone
    directory: a clone made for another branch is removed and cloned again, one for the
    same branch is fetched and reset, which follows a force push. A branch gone from the
    remote raises. A local path rejects ``branch``.

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

    auth_env = _credential_env(credentials) if credentials else None

    base = Path(clones_dir) if clones_dir else default_clones_dir()
    target = base / _clone_slug(clean_url)
    # The slug regex already forbids separators and dot-runs; keep an
    # explicit containment check so a clone can never land outside the
    # clones directory regardless of what the URL decomposed into.
    base_real = os.path.realpath(base)
    if not os.path.realpath(target).startswith(base_real.rstrip(os.sep) + os.sep):
        raise CodeRepositoryError(message=f"Could not derive a safe clone name for {clean_url}.")

    def _branch_not_found() -> CodeRepositoryError:
        return CodeRepositoryError(message=f"Branch '{branch}' was not found in '{clean_url}'.")

    async with _clone_lock(target):
        # A directory without .git is not a clone (a clone that was never completed),
        # and a clone made for another branch must not be refreshed under this
        # request: both fall through to a new clone that replaces them.
        is_clone = (target / ".git").is_dir()
        if is_clone and await asyncio.to_thread(read_clone_branch, target) == branch:
            # Legacy URL-embedded credentials: fetch from the explicit URL
            # instead of the stored remote (which is deliberately
            # credential-free). Out-of-band credentials ride the environment.
            if branch is None:
                pull_args = ["pull", "--ff-only"] + ([url] if has_credentials else [])
                returncode, stderr = await _run_git(pull_args, cwd=target, env=auth_env)
                refresh = "git pull"
            else:
                # Fetch and reset, not pull: a rebased or force-pushed branch is not a
                # fast-forward. The ref is spelled out because git resolves a bare name
                # itself (a tag before a branch, '+x' as a force refspec) and could land
                # on another commit than 'clone --branch'.
                remote = url if has_credentials else "origin"
                returncode, stderr = await _run_git(
                    ["fetch", "--depth", "1", remote, f"refs/heads/{branch}"],
                    cwd=target,
                    env=auth_env,
                )
                refresh = "git fetch"
                if returncode != 0 and await _remote_ref_missing(url, branch, auth_env):
                    raise _branch_not_found()
                if returncode == 0:
                    returncode, stderr = await _run_git(
                        ["reset", "--hard", "FETCH_HEAD"], cwd=target
                    )
                    refresh = "git reset"
            if returncode != 0:
                # A stale clone is still indexable; the caller asked for the repo,
                # not for freshness guarantees.
                logger.warning(
                    "Reusing existing clone at %s (%s failed: %s)",
                    target,
                    refresh,
                    _scrub(stderr[-500:]),
                )
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            # A clone made for another branch, or what an interrupted clone left behind.
            logger.info(
                "Replacing the clone at %s: it is not a clone of %s",
                target,
                branch or "the default branch",
            )
            await asyncio.to_thread(_remove_tree, target)
        logger.info("Cloning %s on %s into %s", clean_url, branch or "the default branch", target)
        # core.symlinks=false: git otherwise materializes symlinks committed in the
        # remote, and nothing re-validates paths under the clone afterwards -- a repo
        # containing '.enola -> ~/.ssh' would have the enola snapshot written through it,
        # and 'mod.py -> /proc/self/environ' would be read into the code graph. With this
        # set, git writes each symlink as a regular file containing its target path.
        # Passed via -c on clone so it is persisted into the new repo's config and the
        # later 'git pull --ff-only' honours it too.
        clone_args = ["clone", "-c", "core.symlinks=false", "--depth", "1"]
        if branch is not None:
            clone_args += ["--branch", branch]
        returncode, stderr = await _run_git(clone_args + [url, str(target)], env=auth_env)
        if returncode != 0:
            if branch is not None and await _remote_ref_missing(url, branch, auth_env):
                raise _branch_not_found()
            raise CodeRepositoryError(
                message=f"Failed to clone '{clean_url}': {_scrub(stderr[-1000:])}"
            )
        if branch is not None and await asyncio.to_thread(read_clone_branch, target) != branch:
            # 'clone --branch' also accepts a tag, which leaves HEAD detached.
            await asyncio.to_thread(_remove_tree, target)
            raise CodeRepositoryError(message=f"'{branch}' in '{clean_url}' is not a branch.")
        if has_credentials:
            await _run_git(["remote", "set-url", "origin", clean_url], cwd=target)
        return target
