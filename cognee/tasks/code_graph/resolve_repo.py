"""Resolve repository specs (local paths or remote git URLs) to local clones.

Used by ``remember(..., content_type="code")`` so callers can pass a GitHub
URL (or a list of them) and get the enola code-graph pipeline run on a local
shallow clone. Clones live under ``~/.cognee/repos`` and are reused across
calls; an existing clone is refreshed with a best-effort ``git pull``.
"""

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union

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


def _clone_slug(url: str) -> str:
    """A stable directory name for a remote URL, e.g. 'github.com-org-repo'."""
    tail = url.split("://")[-1].replace(":", "/").rstrip("/")
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.")


async def _run_git(args, cwd: Optional[Path] = None) -> tuple:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise CodeRepositoryError(
            message="git is required to clone remote repositories but was not found on PATH."
        )
    process = await asyncio.create_subprocess_exec(
        git_binary,
        *args,
        cwd=str(cwd) if cwd else None,
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
) -> Path:
    """Return a local directory for the repo spec, shallow-cloning remote URLs.

    Local paths are validated and returned as-is. Remote URLs are cloned with
    ``--depth 1`` into ``clones_dir`` (default ``~/.cognee/repos``); an
    existing clone is reused after a best-effort ``git pull --ff-only``.
    Remote resolution honors ``ALLOW_HTTP_REQUESTS=false``.
    """
    if not is_remote_repo(spec):
        path = Path(spec).expanduser()
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
    target = Path(clones_dir) if clones_dir else DEFAULT_CLONES_DIR
    target = target / _clone_slug(url)

    if (target / ".git").is_dir():
        returncode, stderr = await _run_git(["pull", "--ff-only"], cwd=target)
        if returncode != 0:
            # A stale clone is still indexable; the caller asked for the repo,
            # not for freshness guarantees.
            logger.warning(
                "Reusing existing clone at %s (git pull failed: %s)", target, stderr[-500:]
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", url, target)
    returncode, stderr = await _run_git(["clone", "--depth", "1", url, str(target)])
    if returncode != 0:
        raise CodeRepositoryError(message=f"Failed to clone '{url}': {stderr[-1000:]}")
    return target
