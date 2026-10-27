"""Disk admission and monitoring for managed repository clones and their output.

The operator supplies the budget; the SDK does not impose a repository-size limit.
Polling is a guard, not a filesystem quota. Cloud must reserve headroom and enforce
the volume's hard quota separately. All clones under one root count toward the budget.
"""

import asyncio
import errno
import os
import shutil
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from weakref import WeakSet

from fastapi import status
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.exceptions import CogneeConfigurationError
from cognee.tasks.code_graph.process import finish_cleanup
from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

_DISK_FULL_MESSAGES = (b"no space left on device", b"disk quota exceeded")


def reports_disk_full(stderr: bytes) -> bool:
    """Whether a child process's stderr says the filesystem is full, in any letter case.

    git capitalises the message, enola (Go) prints it in lowercase.
    """
    lowered = stderr.lower()
    return any(message in lowered for message in _DISK_FULL_MESSAGES)


class RepositoryResourceError(CodeRepositoryError):
    """Measured storage exceeded a configured budget, or the filesystem refused a write.

    Physical ENOSPC/EDQUOT failures have no configured limit or byte measurement.
    """

    def __init__(
        self,
        reason: str,
        limit_bytes: int | None = None,
        used_bytes: int | None = None,
        free_bytes: int | None = None,
    ):
        self.reason = reason
        self.limit_bytes = limit_bytes
        self.used_bytes = used_bytes
        self.free_bytes = free_bytes
        if limit_bytes is None:
            measured = "filesystem refused the write"
        elif used_bytes is not None:
            measured = f"{used_bytes} bytes used; budget {limit_bytes} bytes"
        else:
            measured = f"{free_bytes} bytes free; at least {limit_bytes} bytes required"
        if limit_bytes is None:
            remedy = "Free disk space or use a worker with more space."
        elif used_bytes is not None:
            remedy = (
                "Raise COGNEE_REPOS_MAX_BYTES or free space under the clones and data directories."
            )
        else:
            remedy = "Free disk space, or lower COGNEE_REPOS_MIN_FREE_BYTES."
        super().__init__(
            message=f"Repository processing stopped: {reason} ({measured}). {remedy}",
            name="RepositoryResourceError",
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
        )


class RepositoryDiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COGNEE_REPOS_", extra="ignore")

    max_bytes: int | None = Field(default=None, gt=0)
    min_free_bytes: int | None = Field(default=None, gt=0)

    @field_validator("max_bytes", "min_free_bytes", mode="before")
    @classmethod
    def _empty_is_unset(cls, value):
        """A variable left empty, as .env.template shows it, means no limit."""
        return None if isinstance(value, str) and not value.strip() else value


def repository_disk_config() -> RepositoryDiskConfig:
    try:
        return RepositoryDiskConfig()
    except ValidationError as error:
        raise CogneeConfigurationError(
            message="COGNEE_REPOS_MAX_BYTES and COGNEE_REPOS_MIN_FREE_BYTES must each be a "
            "positive integer number of bytes, for example 5000000000 (not 5GB)."
        ) from error


def stat_disk_bytes(info: os.stat_result) -> int:
    """What a file occupies: its size, or its allocated blocks when those are more."""
    return max(info.st_size, getattr(info, "st_blocks", 0) * 512)


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
                    except (FileNotFoundError, PermissionError):
                        # Git renamed or removed a temporary pack during this scan, or the
                        # entry is not ours to stat (a volume's lost+found).
                        continue
                    total += stat_disk_bytes(info)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
        except (FileNotFoundError, PermissionError):
            # A completed/failed clone can disappear during the scan, and a directory we
            # may not read (a volume's lost+found) holds nothing of ours.
            continue
    return total


def _check_free_space(root: Path, config: RepositoryDiskConfig) -> None:
    if config.min_free_bytes is not None:
        existing = root
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free <= config.min_free_bytes:
            raise RepositoryResourceError(
                "insufficient free disk space", config.min_free_bytes, free_bytes=free
            )


def repository_storage_roots(clone_root: Path) -> list[Path]:
    """Count managed clones and local durable document storage, once per subtree."""
    from cognee.infrastructure.files.storage import get_storage_config

    roots = [clone_root.resolve()]
    storage = get_storage_config()["data_root_directory"]
    if not str(storage).startswith("s3://"):
        from cognee.infrastructure.files.storage.LocalFileStorage import get_parsed_path

        roots.append(Path(get_parsed_path(str(storage))).expanduser().resolve())
    return [
        root
        for root in set(roots)
        if not any(root != other and root.is_relative_to(other) for other in roots)
    ]


def _check_repository_free_space(roots: list[Path], config: RepositoryDiskConfig) -> None:
    for root in roots:
        _check_free_space(root, config)


def _check_repository_size(roots: list[Path], config: RepositoryDiskConfig) -> int | None:
    """Scan managed storage and raise when its configured budget is reached."""
    used = None
    if config.max_bytes is not None:
        used = sum(disk_usage_bytes(root) for root in roots)
        if used >= config.max_bytes:
            raise RepositoryResourceError(
                "repository storage budget exceeded", config.max_bytes, used_bytes=used
            )
    return used


def check_repository_disk(roots: list[Path], config: RepositoryDiskConfig) -> int | None:
    """Check free space first, then return managed bytes used when configured."""
    _check_repository_free_space(roots, config)
    return _check_repository_size(roots, config)


class DocumentCopyBudget:
    """Admit known-size document copies before a synchronous storage write can fill the disk.

    The storage is scanned once, when the budget is created; each admitted copy is then
    added to that baseline, so a repository with thousands of documents costs one scan.
    """

    def __init__(self, clone_root: Path):
        from cognee.infrastructure.files.storage import get_storage_config

        self._config = repository_disk_config()
        self._active = self._config.max_bytes is not None or self._config.min_free_bytes is not None
        self._destination = None
        self._used = 0
        if not self._active:
            return
        roots = repository_storage_roots(clone_root)
        self._used = check_repository_disk(roots, self._config) or 0
        storage = get_storage_config()["data_root_directory"]
        if not str(storage).startswith("s3://"):
            from cognee.infrastructure.files.storage.LocalFileStorage import get_parsed_path

            self._destination = Path(get_parsed_path(str(storage))).expanduser().resolve()

    def admit(self, size: int) -> None:
        if self._destination is None:
            return
        config = self._config
        if config.max_bytes is not None and self._used + size >= config.max_bytes:
            raise RepositoryResourceError(
                "document copy exceeds repository storage budget",
                config.max_bytes,
                used_bytes=self._used + size,
            )
        if config.min_free_bytes is not None:
            existing = self._destination
            while not existing.exists():
                existing = existing.parent
            free = shutil.disk_usage(existing).free
            if free - size <= config.min_free_bytes:
                raise RepositoryResourceError(
                    "document copy leaves insufficient free disk space",
                    config.min_free_bytes,
                    free_bytes=free - size,
                )
        self._used += size


# Tasks that already run under a budget: a nested budget must not start a second monitor.
_ACTIVE_BUDGETS: WeakSet = WeakSet()

# Size scans pause for 10 scan times (at least 0.1 s), using about a tenth of one thread.
# Free-space checks run every 0.1 s, on the event loop: they are a few syscalls, and a thread
# would queue behind blocked scans in the shared executor.
_MONITOR_MIN_PAUSE_SECONDS = 0.1
_MONITOR_PAUSE_PER_SCAN = 10


@asynccontextmanager
async def repository_disk_budget(clone_root: Path):
    """Apply configured guards and classify filesystem disk exhaustion as HTTP 507."""
    try:
        async with _monitored_repository_disk_budget(clone_root):
            yield
    except OSError as error:
        if error.errno not in {errno.ENOSPC, errno.EDQUOT}:
            raise
        raise RepositoryResourceError("filesystem space or quota exhausted") from error


@asynccontextmanager
async def _monitored_repository_disk_budget(clone_root: Path):
    """Monitor storage while keeping repository work in the task that owns its lock.

    A polling guard can overshoot. The deployment must reserve headroom below its
    filesystem quota. Cancellation waits for owned subprocess cleanup.
    """
    owner = asyncio.current_task()
    if owner in _ACTIVE_BUDGETS:
        yield
        return
    config = repository_disk_config()
    if config.max_bytes is None and config.min_free_bytes is None:
        yield
        return
    roots = repository_storage_roots(clone_root)
    await asyncio.to_thread(check_repository_disk, roots, config)
    _ACTIVE_BUDGETS.add(owner)
    failure = []

    async def monitor(check, *, scan: bool):
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            try:
                if scan:
                    await asyncio.to_thread(check, roots, config)
                else:
                    check(roots, config)
            except (RepositoryResourceError, OSError) as error:
                if not failure:
                    failure.append(error)
                    owner.cancel()
                return
            pause = _MONITOR_MIN_PAUSE_SECONDS
            if scan:
                pause = max(pause, _MONITOR_PAUSE_PER_SCAN * (loop.time() - started))
            await asyncio.sleep(pause)

    watchers = []
    if config.min_free_bytes is not None:
        watchers.append(asyncio.create_task(monitor(_check_repository_free_space, scan=False)))
    if config.max_bytes is not None:
        watchers.append(asyncio.create_task(monitor(_check_repository_size, scan=True)))
    try:
        yield
        await asyncio.to_thread(check_repository_disk, roots, config)
    except asyncio.CancelledError:
        if failure:
            uncancel = getattr(owner, "uncancel", None)
            if callable(uncancel):
                uncancel()
                if owner.cancelling():
                    # The owner was cancelled from outside too: that cancellation wins.
                    raise
            raise failure[0] from None
        raise
    finally:
        _ACTIVE_BUDGETS.discard(owner)
        for watcher in watchers:
            watcher.cancel()
        await finish_cleanup(asyncio.gather(*watchers, return_exceptions=True))
