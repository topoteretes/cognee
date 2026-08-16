"""Persistence for user-authorized external database connections.

CRUD over the ``tool_connections`` table plus the deployment-level
connections configured via ``TOOL_SQL_CONNECTIONS``. Connection strings are
encrypted at rest through :mod:`cognee.modules.integrations.crypto` and only
decrypted by :func:`get_tool_connection` at the moment a query runs.

Resolution **fails closed**: a name that doesn't exist for the acting user
raises :class:`ToolConnectionNotFoundError`, whether the row is missing or
owned by someone else.
"""

import asyncio
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.crypto import decrypt_credentials, encrypt_credentials
from cognee.modules.tools.config import get_tools_config
from cognee.modules.tools.errors import ToolConnectionNotFoundError, ToolError
from cognee.modules.tools.models.ToolConnection import ToolConnection
from cognee.modules.tools.target_policy import validate_user_connection_target
from cognee.shared.logging_utils import get_logger

logger = get_logger("tool_connections")

STATUS_ACTIVE = "active"

# Providers the text-to-SQL executor supports in v1.
_SUPPORTED_PROVIDERS = ("postgres", "sqlite")

_table_ready = False
_table_lock = asyncio.Lock()


def infer_provider(connection_string: str) -> str:
    """Infer the provider from a DSN scheme; raise for unsupported ones."""
    scheme = connection_string.split("://", 1)[0].lower() if "://" in connection_string else ""
    if scheme.startswith("postgres"):
        return "postgres"
    if scheme.startswith("sqlite"):
        return "sqlite"
    raise ToolError(
        f"Unsupported tool connection provider for scheme '{scheme or connection_string[:20]}'. "
        f"Supported providers: {', '.join(_SUPPORTED_PROVIDERS)}."
    )


async def _ensure_table() -> None:
    """Create the tool_connections table on first use.

    Deployments that predate this table never ran a setup step that knows
    about it, so the registration path creates it lazily (idempotent
    CREATE TABLE IF NOT EXISTS semantics via checkfirst).
    """
    global _table_ready
    if _table_ready:
        return
    async with _table_lock:
        if _table_ready:
            return
        from cognee.modules.tools.models.ToolWriteProposal import ToolWriteProposal

        engine = get_relational_engine()
        async with engine.engine.begin() as connection:
            for table in (ToolConnection.__table__, ToolWriteProposal.__table__):
                await connection.run_sync(
                    lambda sync_connection, table=table: table.create(
                        sync_connection, checkfirst=True
                    )
                )
        _table_ready = True


def _row_to_public(row: ToolConnection) -> dict[str, Any]:
    """Non-secret view of a connection — never includes the DSN."""
    options = row.options or {}
    return {
        "name": row.name,
        "provider": row.provider,
        "allowed_tables": options.get("allowed_tables"),
        "max_rows": options.get("max_rows"),
        "description": options.get("description"),
        "allow_writes": bool(options.get("allow_writes")),
        "origin": "user",
    }


def _env_connections() -> dict[str, dict[str, Any]]:
    config = get_tools_config()
    try:
        return config.sql_connections()
    except (ValueError, TypeError) as error:
        logger.error("Invalid TOOL_SQL_CONNECTIONS configuration: %s", error)
        raise


async def register_tool_connection(
    user_id: UUID,
    name: str,
    connection_string: str,
    *,
    provider: Optional[str] = None,
    allowed_tables: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
    description: Optional[str] = None,
    allow_writes: bool = False,
) -> dict[str, Any]:
    """Insert or replace the caller's connection under ``name``.

    The DSN is encrypted before it touches the database; registration fails
    with the crypto module's error when no encryption key is configured —
    there is deliberately no plaintext fallback.

    ``allow_writes`` opts this connection into approval-gated write-back
    (correction proposals). Reads never need it.
    """
    if not name or not name.strip():
        raise ToolError("Tool connection name must be a non-empty string")
    resolved_provider = provider or infer_provider(connection_string)
    if resolved_provider not in _SUPPORTED_PROVIDERS:
        raise ToolError(
            f"Unsupported tool connection provider '{resolved_provider}'. "
            f"Supported providers: {', '.join(_SUPPORTED_PROVIDERS)}."
        )

    config = get_tools_config()
    validate_user_connection_target(
        connection_string,
        resolved_provider,
        sqlite_allowed_root=config.tool_sqlite_allowed_root,
        allowed_hosts=config.sql_allowed_hosts(),
    )

    ciphertext, nonce, encryption_version, key_id = encrypt_credentials(
        {"connection_string": connection_string}
    )
    options = {
        "allowed_tables": allowed_tables,
        "max_rows": max_rows,
        "description": description,
        "allow_writes": allow_writes,
    }

    await _ensure_table()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        result = await session.execute(
            select(ToolConnection).where(
                ToolConnection.user_id == user_id, ToolConnection.name == name
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = ToolConnection(user_id=user_id, name=name)
            session.add(row)

        row.provider = resolved_provider
        row.ciphertext = ciphertext
        row.nonce = nonce
        row.encryption_version = encryption_version
        row.key_id = key_id
        row.options = options
        row.status = STATUS_ACTIVE

        await session.commit()
        await session.refresh(row)
        return _row_to_public(row)


async def list_tool_connections(user_id: UUID) -> list[dict[str, Any]]:
    """The caller's registered connections plus deployment-level ones.

    Never returns connection strings.
    """
    connections: list[dict[str, Any]] = []

    await _ensure_table()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        result = await session.execute(
            select(ToolConnection)
            .where(ToolConnection.user_id == user_id, ToolConnection.status == STATUS_ACTIVE)
            .order_by(ToolConnection.name)
        )
        connections.extend(_row_to_public(row) for row in result.scalars().all())

    taken = {connection["name"] for connection in connections}
    for name, options in _env_connections().items():
        # A user-registered connection shadows an env one with the same name.
        if name in taken:
            continue
        connections.append(
            {
                "name": name,
                "provider": options.get("provider") or infer_provider(options["connection_string"]),
                "allowed_tables": options.get("allowed_tables"),
                "max_rows": options.get("max_rows"),
                "description": options.get("description"),
                "origin": "env",
            }
        )
    return connections


async def get_tool_connection(user_id: UUID, name: str) -> dict[str, Any]:
    """Resolve a connection to its full config including the decrypted DSN.

    Call only at query-execution time. Fails closed with
    :class:`ToolConnectionNotFoundError` for unknown or non-owned names.
    """
    await _ensure_table()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        result = await session.execute(
            select(ToolConnection).where(
                ToolConnection.user_id == user_id,
                ToolConnection.name == name,
                ToolConnection.status == STATUS_ACTIVE,
            )
        )
        row = result.scalar_one_or_none()

    if row is not None:
        payload = decrypt_credentials(row.ciphertext, row.nonce, row.encryption_version, row.key_id)
        config = get_tools_config()
        validate_user_connection_target(
            payload["connection_string"],
            row.provider,
            sqlite_allowed_root=config.tool_sqlite_allowed_root,
            allowed_hosts=config.sql_allowed_hosts(),
        )
        options = row.options or {}
        return {
            "name": row.name,
            "provider": row.provider,
            "connection_string": payload["connection_string"],
            "allowed_tables": options.get("allowed_tables"),
            "max_rows": options.get("max_rows"),
            "allow_writes": bool(options.get("allow_writes")),
        }

    env_connection = _env_connections().get(name)
    if env_connection is not None:
        return {
            "name": name,
            "provider": env_connection.get("provider")
            or infer_provider(env_connection["connection_string"]),
            "connection_string": env_connection["connection_string"],
            "allowed_tables": env_connection.get("allowed_tables"),
            "max_rows": env_connection.get("max_rows"),
            "allow_writes": bool(env_connection.get("allow_writes")),
        }

    raise ToolConnectionNotFoundError(
        f"No authorized tool connection named '{name}' for this user."
    )


async def delete_tool_connection(user_id: UUID, name: str) -> bool:
    """Remove the caller's connection. Returns False when nothing matched."""
    await _ensure_table()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        result = await session.execute(
            select(ToolConnection).where(
                ToolConnection.user_id == user_id, ToolConnection.name == name
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
        return True
