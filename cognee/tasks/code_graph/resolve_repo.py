"""Resolve repository specs (local paths or remote git URLs) to local clones.

Used by ``remember(..., content_type="code")`` so callers can pass a GitHub
URL (or a list of them) and get the enola code-graph pipeline run on a local
shallow clone. Clones live under ``~/.cognee/repos`` and are reused across
calls; an existing clone is refreshed with a best-effort ``git pull``.
"""

import asyncio
import base64
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlsplit, urlunsplit

from fastapi import status

from cognee.exceptions import CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("code_graph")

_REMOTE_PREFIXES = ("https://", "http://", "git@", "ssh://")

_FALSEY = {"false", "0", "no", "off"}

_GIT_TIMEOUT_SECONDS = 600

DEFAULT_CLONES_DIR = Path.home() / ".cognee" / "repos"


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


def redact_repo_spec(spec: Union[str, Path]) -> str:
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
    """A stable directory name for a remote URL, e.g. 'github.com-org-repo'."""
    tail = url.split("://")[-1].replace(":", "/").rstrip("/")
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.")


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


async def _run_git(args, cwd: Optional[Path] = None, env: Optional[dict] = None) -> tuple:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise CodeRepositoryError(
            message="git is required to clone remote repositories but was not found on PATH."
        )
    process = await asyncio.create_subprocess_exec(
        git_binary,
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


async def resolve_repo_source(
    spec: Union[str, Path],
    clones_dir: Optional[Path] = None,
    credentials: Optional[str] = None,
) -> Path:
    """Return a local directory for the repo spec, shallow-cloning remote URLs.

    Local paths are validated and returned as-is. Remote URLs are cloned with
    ``--depth 1`` into ``clones_dir`` (default ``~/.cognee/repos``); an
    existing clone is reused after a best-effort ``git pull --ff-only``.
    Remote resolution honors ``ALLOW_HTTP_REQUESTS=false``.

    ``credentials`` is the preferred way to authenticate against a private
    https remote (e.g. a GitHub App installation token): it reaches git only
    through environment-level config (see :func:`_credential_env`), so the
    URL — and everything derived from it (slug, remote, logs, errors) —
    never carries a secret. Embedding credentials in the URL userinfo still
    works but is the legacy path.
    """
    if not is_remote_repo(spec):
        from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path

        # Repo specs can arrive from outside the SDK (CLI arguments, API
        # callers), so local paths take the same allowlist containment check
        # as ingestion's local-file reads instead of dereferencing an
        # arbitrary path.
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

    base = Path(clones_dir) if clones_dir else DEFAULT_CLONES_DIR
    target = base / _clone_slug(clean_url)
    # The slug regex already forbids separators and dot-runs; keep an
    # explicit containment check so a clone can never land outside the
    # clones directory regardless of what the URL decomposed into.
    base_real = os.path.realpath(base)
    if not os.path.realpath(target).startswith(base_real.rstrip(os.sep) + os.sep):
        raise CodeRepositoryError(message=f"Could not derive a safe clone name for {clean_url}.")

    if (target / ".git").is_dir():
        # Legacy URL-embedded credentials: fetch from the explicit URL
        # instead of the stored remote (which is deliberately
        # credential-free). Out-of-band credentials ride the environment.
        pull_args = ["pull", "--ff-only"] + ([url] if has_credentials else [])
        returncode, stderr = await _run_git(pull_args, cwd=target, env=auth_env)
        if returncode != 0:
            # A stale clone is still indexable; the caller asked for the repo,
            # not for freshness guarantees.
            logger.warning(
                "Reusing existing clone at %s (git pull failed: %s)", target, _scrub(stderr[-500:])
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", clean_url, target)
    # core.symlinks=false: git otherwise materializes symlinks committed in the
    # remote, and nothing re-validates paths under the clone afterwards -- a repo
    # containing '.enola -> ~/.ssh' would have the enola snapshot written through it,
    # and 'mod.py -> /proc/self/environ' would be read into the code graph. With this
    # set, git writes each symlink as a regular file containing its target path.
    # Passed via -c on clone so it is persisted into the new repo's config and the
    # later 'git pull --ff-only' honours it too.
    returncode, stderr = await _run_git(
        ["clone", "-c", "core.symlinks=false", "--depth", "1", url, str(target)], env=auth_env
    )
    if returncode != 0:
        raise CodeRepositoryError(
            message=f"Failed to clone '{clean_url}': {_scrub(stderr[-1000:])}"
        )
    if has_credentials:
        await _run_git(["remote", "set-url", "origin", clean_url], cwd=target)
    return target
