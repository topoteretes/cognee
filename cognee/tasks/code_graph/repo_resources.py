"""Disk admission and monitoring for managed repository clones and their output.

The operator supplies the budget; the SDK does not impose a repository-size limit.
Polling is a guard, not a filesystem quota. Cloud must reserve headroom and enforce
the volume's hard quota separately. All clones under one root count toward the budget.
"""

import asyncio
import os
import shutil
import stat
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.exceptions import CogneeConfigurationError, CogneeSystemError


class RepositoryResourceError(CogneeSystemError):
    """The clone root is over its byte budget (``used_bytes``) or short of free space
    (``free_bytes``); exactly one of the two is set, as the budget asks."""

    def __init__(
        self,
        reason: str,
        limit_bytes: int,
        used_bytes: int | None = None,
        free_bytes: int | None = None,
    ):
        self.reason = reason
        self.limit_bytes = limit_bytes
        self.used_bytes = used_bytes
        self.free_bytes = free_bytes
        if used_bytes is not None:
            measured = f"{used_bytes} bytes used; budget {limit_bytes} bytes"
        else:
            measured = f"{free_bytes} bytes free; at least {limit_bytes} bytes required"
        super().__init__(
            message=f"Repository processing stopped: {reason} ({measured}). "
            "Use a worker with more space or ingest a prepared local checkout.",
            name="RepositoryResourceError",
            status_code=507,
        )


class RepositoryDiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COGNEE_REPOS_", extra="ignore")

    max_bytes: int | None = Field(default=None, gt=0)
    min_free_bytes: int | None = Field(default=None, gt=0)


def repository_disk_config() -> RepositoryDiskConfig:
    try:
        return RepositoryDiskConfig()
    except ValidationError as error:
        raise CogneeConfigurationError(
            message="COGNEE_REPOS_MAX_BYTES and COGNEE_REPOS_MIN_FREE_BYTES must each be a "
            "positive integer number of bytes, for example 5000000000 (not 5GB)."
        ) from error


def disk_usage_bytes(root: Path) -> int:
    """Count files, directories and temporary output without following symlinks."""
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue  # Git renamed or removed a temporary pack during this scan.
                    total += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
        except FileNotFoundError:
            continue  # A completed/failed clone can disappear during the scan.
    return total


def check_disk(root: Path, config: RepositoryDiskConfig) -> None:
    if config.max_bytes is not None:
        used = disk_usage_bytes(root)
        if used >= config.max_bytes:
            raise RepositoryResourceError(
                "clone storage budget exceeded", config.max_bytes, used_bytes=used
            )
    if config.min_free_bytes is not None:
        existing = root
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free <= config.min_free_bytes:
            raise RepositoryResourceError(
                "insufficient free disk space", config.min_free_bytes, free_bytes=free
            )


async def with_repository_disk_budget(operation, root: Path):
    """Run an async callable, cancelling and awaiting its cleanup on disk exhaustion."""
    from cognee.tasks.code_graph.resolve_repo import finish_cleanup

    config = repository_disk_config()
    if config.max_bytes is None and config.min_free_bytes is None:
        return await operation()
    await asyncio.to_thread(check_disk, root, config)

    async def monitor():
        # The scan walks the whole clone root, so pause ten times as long as it took:
        # the guard then uses about a tenth of one worker thread, however big the root.
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.to_thread(check_disk, root, config)
            await asyncio.sleep(max(0.1, 10 * (loop.time() - started)))

    running = asyncio.create_task(operation())
    watching = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait({running, watching}, return_when=asyncio.FIRST_COMPLETED)
        if watching in done:
            await watching
        result = await running
        await asyncio.to_thread(check_disk, root, config)
        return result
    finally:
        running.cancel()
        watching.cancel()
        await finish_cleanup(asyncio.gather(running, watching, return_exceptions=True))
