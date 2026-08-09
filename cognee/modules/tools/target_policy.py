"""Deployment policy checks for user-supplied SQL connection targets."""

from pathlib import Path
from urllib.parse import urlsplit

from cognee.modules.tools.errors import ToolError

_SQLITE_SCHEMES = ("sqlite://", "sqlite+aiosqlite://")


def validate_user_connection_target(
    connection_string: str,
    provider: str,
    *,
    sqlite_allowed_root: str,
    allowed_hosts: set[str],
) -> None:
    """Reject user-supplied database targets outside deployment allowlists.

    Deployment-level connections are already administrator supplied and do
    not use this function. An empty allowlist intentionally rejects all
    corresponding user targets.
    """
    if provider == "sqlite":
        if not sqlite_allowed_root:
            raise ToolError(
                "User SQLite connections are disabled. Configure "
                "TOOL_SQLITE_ALLOWED_ROOT with an approved directory."
            )

        matching_scheme = next(
            (scheme for scheme in _SQLITE_SCHEMES if connection_string.startswith(scheme)),
            None,
        )
        if matching_scheme is None:
            raise ToolError("Unrecognized SQLite connection string scheme.")

        path = connection_string[len(matching_scheme) :]
        # URI forms can address arbitrary files through query parameters or
        # SQLite virtual filenames. Only ordinary filesystem paths are safe to
        # constrain to the configured root.
        if not path or path.startswith("/file:") or "?" in path or "#" in path:
            raise ToolError(
                "User SQLite connections must be ordinary paths under "
                "TOOL_SQLITE_ALLOWED_ROOT."
            )

        root = Path(sqlite_allowed_root).expanduser().resolve()
        target = (
            Path(path[1:]).expanduser().resolve()
            if path.startswith("//")
            else (root / path.lstrip("/")).resolve()
        )
        if target == root or root not in target.parents:
            raise ToolError(
                "User SQLite connection must point inside "
                "TOOL_SQLITE_ALLOWED_ROOT."
            )
        return

    if provider == "postgres":
        parsed = urlsplit(connection_string)
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host or host not in {entry.rstrip(".") for entry in allowed_hosts}:
            raise ToolError(
                "User PostgreSQL connections are restricted to hosts listed in "
                "TOOL_SQL_ALLOWED_HOSTS."
            )
        return

    raise ToolError(f"Unsupported tool connection provider: {provider}")
