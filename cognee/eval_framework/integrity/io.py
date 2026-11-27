import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cognee.eval_framework.integrity.errors import RunIntegrityError


def _fsync_directory(directory: str) -> None:
    """Make a rename durable. Directories cannot be opened for fsync on Windows."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str, payload: Any, indent: int = 4) -> None:
    """Write JSON so readers see either the previous file or the complete new one.

    The payload goes to a temporary file in the same directory, is fsynced, then renamed over
    the target and the directory is fsynced. On any failure the temporary file is removed and
    the previous file is untouched.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed below; delete=False keeps the path
        "w", encoding="utf-8", dir=directory, prefix=".tmp-", suffix=".json", delete=False
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
    _fsync_directory(directory)


LOCK_FILENAME = "run.lock"


@contextmanager
def run_lock(run_dir: str) -> Iterator[None]:
    """Hold an exclusive lock on a run directory, so two runners never write the same
    artifacts. The lock is a file created with O_EXCL and removed on exit. A runner killed
    without cleanup leaves it behind; it is never taken over automatically, remove it by hand
    once you are sure no run is active."""
    lock_path = os.path.join(run_dir or ".", LOCK_FILENAME)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunIntegrityError(
            f"Run directory is locked by {lock_path}. Another eval run is using it; if none is "
            "running (a previous run was killed), delete the file and retry."
        ) from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        os.remove(lock_path)


def load_json_list(path: str, artifact: str) -> list[dict[str, Any]]:
    """Load a run artifact that must be a JSON list of objects.

    Raises RunIntegrityError for a missing, truncated or wrongly shaped file, naming the file
    so the operator knows which artifact to regenerate.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise RunIntegrityError(f"{artifact} file not found: {path}") from None
    except json.JSONDecodeError as error:
        raise RunIntegrityError(
            f"{artifact} file {path} is not valid JSON (truncated or corrupt): {error}"
        ) from error
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise RunIntegrityError(f"{artifact} file {path} must contain a JSON list of objects")
    return data
