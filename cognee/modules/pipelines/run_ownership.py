"""OS-backed ownership for safe recovery on a shared system directory.

Age makes a run eligible for recovery; acquiring its existing ownership lock
proves the writer has released it or died. Missing markers (including legacy
rows and workers with a different filesystem) are deliberately unverifiable.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from filelock import FileLock, Timeout

from cognee.base_config import get_base_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("pipeline.run_ownership")


def _lock_path(token: str) -> Path:
    # Only UUIDs from our run records can select a lock, never arbitrary paths.
    return Path(get_base_config().system_root_directory) / "pipeline_run_locks" / str(UUID(token))


@dataclass
class RunOwnership:
    token: str | None
    closed: bool = False


@contextmanager
def pipeline_run_ownership():
    owner = RunOwnership(str(uuid4()))
    path = _lock_path(owner.token)
    try:
        path.mkdir(parents=True, exist_ok=True)
        lock = FileLock(path / "owner.lock", thread_local=False)
        lock.acquire(timeout=0)
    except OSError:
        logger.warning(
            "Run ownership unavailable; automatic recovery disabled for this run", exc_info=True
        )
        yield RunOwnership(None)
        return
    try:
        yield owner
    finally:
        lock.release()
        if owner.closed:
            _remove_marker(path)


@contextmanager
def claim_run_ownership(run):
    """Nonblocking recovery claim; no marker or a live writer means no claim."""
    token = (run.run_info or {}).get("recovery_lock")
    try:
        path = _lock_path(token) if isinstance(token, str) else None
    except ValueError:
        path = None
    if path is None or not path.is_dir():
        yield None
        return
    lock = FileLock(path / "owner.lock", thread_local=False)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        yield None
        return
    owner = RunOwnership(token)
    try:
        yield owner
    finally:
        lock.release()
        if owner.closed:
            _remove_marker(path)


def _remove_marker(path: Path) -> None:
    # Release before removing: Windows cannot unlink an open lock file. A
    # competing recovery must re-read the terminal record before any rollback.
    try:
        (path / "owner.lock").unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        logger.debug("Completed-run ownership marker cleanup deferred", exc_info=True)
