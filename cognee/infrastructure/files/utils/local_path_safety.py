import os
from pathlib import Path

from cognee.base_config import get_base_config


ALLOWED_LOCAL_FILE_ROOTS_ENV = "COGNEE_ALLOWED_LOCAL_FILE_ROOTS"


def get_allowed_local_file_roots() -> tuple[Path, ...] | None:
    """Roots that local file paths must live under, or ``None`` for no restriction.

    The allowlist is opt-in: it is enforced only when
    ``COGNEE_ALLOWED_LOCAL_FILE_ROOTS`` is set (``os.pathsep``-separated
    directories). Unset — the default — any local path is readable, subject to
    ``ACCEPT_LOCAL_FILE_PATH``; this is what lets a local server ingest a
    repository or document tree from wherever it lives on the machine.
    Deployments that expose the API to untrusted callers should set it.
    """
    configured_roots = os.getenv(ALLOWED_LOCAL_FILE_ROOTS_ENV)
    if not configured_roots:
        return None

    root_values = [root for root in configured_roots.split(os.pathsep) if root]

    # Cognee's own storage directories are always allowed: internal reads of
    # stored data (open_data_file) go through the same allowlist, and a
    # restrictive COGNEE_ALLOWED_LOCAL_FILE_ROOTS must not lock cognee out of
    # its own data, system, cache, log, and repo-clone roots. The clones root
    # holds repositories cognee cloned on request; its document files (README,
    # docs) are ingested by path exactly like a local project's, so the same
    # allowlist check they pass through has to admit it.
    base_config = get_base_config()
    root_values.extend(
        [
            base_config.data_root_directory,
            base_config.system_root_directory,
            base_config.cache_root_directory,
            base_config.logs_root_directory,
            base_config.repos_root_directory,
        ]
    )

    # realpath canonicalizes symlinks (and e.g. macOS /tmp -> /private/tmp) without
    # requiring the path to exist, so containment checks below are symlink-safe.
    return tuple(Path(os.path.realpath(os.path.expanduser(root))) for root in root_values)


def resolve_local_path(path: str | Path, *, must_exist: bool = False) -> Path:
    # realpath follows symlinks, so a path cannot escape an allowed root through a
    # symlink that points outside it (a lexical abspath check would accept such a
    # path). The realpath + startswith containment check below is the idiom static
    # analyzers (CodeQL py/path-injection) recognize as a path sanitizer.
    resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    allowed_roots = get_allowed_local_file_roots()

    if allowed_roots is None:
        resolved_path = Path(resolved)
        if must_exist and not resolved_path.exists():
            raise FileNotFoundError(path)
        return resolved_path

    for root in allowed_roots:
        root_str = os.fspath(root)
        if resolved == root_str:
            resolved_path = Path(root_str)
        elif resolved.startswith(root_str.rstrip(os.sep) + os.sep):
            resolved_path = Path(resolved)
        else:
            continue
        if must_exist and not resolved_path.exists():
            raise FileNotFoundError(path)
        return resolved_path

    raise ValueError("Local file path is outside allowed roots.")
