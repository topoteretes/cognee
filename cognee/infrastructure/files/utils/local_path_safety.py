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
    root_values.extend(_storage_root_directories())

    return _canonical_paths(root_values)


def get_cognee_internal_roots() -> tuple[Path, ...]:
    """Every location holding cognee's own state: the storage roots plus the
    database files, which their settings can move outside those roots.

    Ingestion must never read these as a *source*. They hold every tenant's
    users, ACLs, sessions and stored files, so a caller who could point a
    source at them would read past the multi-tenant boundary.
    """
    # Imported here: the database configs sit above this module in the import graph.
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.relational.config import get_relational_config
    from cognee.infrastructure.databases.vector.config import get_vectordb_config

    locations: list[str | None] = [*_storage_root_directories()]
    locations.append(get_relational_config().db_path)
    locations.append(get_graph_config().graph_file_path)
    locations.extend(
        _local_database_path(url)
        for url in (get_vectordb_config().vector_db_url, get_cache_config().cache_db_url)
    )

    return _canonical_paths([location for location in locations if location])


def is_within_roots(path: str | Path, roots: tuple[Path, ...]) -> bool:
    """True when the canonical ``path`` is one of ``roots`` or lies under one."""
    resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    return any(
        resolved == os.fspath(root) or resolved.startswith(os.fspath(root).rstrip(os.sep) + os.sep)
        for root in roots
    )


def _storage_root_directories() -> list[str]:
    base_config = get_base_config()
    return [
        base_config.data_root_directory,
        base_config.system_root_directory,
        base_config.cache_root_directory,
        base_config.logs_root_directory,
        base_config.repos_root_directory,
    ]


def _local_database_path(value: str | None) -> str | None:
    """The file behind a database setting: a bare path, or the path of a SQLite-family
    URL (sqlite, turso). ``None`` for network URLs, which have no local file."""
    if not value:
        return None
    if "://" not in value:
        return value

    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        url = make_url(value)
    except ArgumentError:
        # Not a SQLAlchemy URL (Neptune's `neptune-graph://<id>`, for one): a
        # network endpoint with no local file behind it.
        return None
    return url.database if url.get_backend_name().startswith("sqlite") else None


def _canonical_paths(paths: list[str]) -> tuple[Path, ...]:
    # realpath canonicalizes symlinks (and e.g. macOS /tmp -> /private/tmp) without
    # requiring the path to exist, so containment checks are symlink-safe.
    return tuple(Path(os.path.realpath(os.path.expanduser(path))) for path in paths)


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
