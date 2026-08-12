"""Async engines for authorized external databases, read-only by construction.

Deliberately NOT :class:`SQLAlchemyAdapter`: that adapter applies write
PRAGMAs (``journal_mode=WAL``) on every SQLite connection and its
``execute_query`` commits — both wrong for a read-only external database.
These engines carry read-only settings at the connection level so the
protection applies to every statement, whatever the SQL guard missed.
"""

from functools import lru_cache

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from cognee.modules.tools.errors import ToolError
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE

_POSTGRES_SCHEME_RE = ("postgres://", "postgresql://", "postgresql+psycopg2://")
_SQLITE_SCHEME_RE = ("sqlite://", "sqlite+aiosqlite://")


def _normalize_url(connection_string: str, provider: str, read_only: bool = True) -> str:
    """Rewrite the DSN onto the async driver; force SQLite into read-only mode.

    ``read_only=False`` is used only by the approval-gated write path
    (apply/dry-run of correction proposals) — it skips the SQLite ``mode=ro``
    URI so the UPDATE can execute.
    """
    if provider == "postgres":
        for scheme in _POSTGRES_SCHEME_RE:
            if connection_string.startswith(scheme):
                return "postgresql+asyncpg://" + connection_string[len(scheme) :]
        if connection_string.startswith("postgresql+asyncpg://"):
            return connection_string
        raise ToolError(f"Unrecognized postgres connection string scheme: {connection_string[:30]}")

    if provider == "sqlite":
        for scheme in _SQLITE_SCHEME_RE:
            if connection_string.startswith(scheme):
                path = connection_string[len(scheme) :]
                break
        else:
            raise ToolError(
                f"Unrecognized sqlite connection string scheme: {connection_string[:30]}"
            )
        if not read_only:
            return (
                f"sqlite+aiosqlite:///{path.lstrip('/') if not path.startswith('//') else path[1:]}"
            )
        if path.startswith("/file:"):
            return f"sqlite+aiosqlite://{path}"
        # sqlite:///relative or sqlite:////absolute — keep the path exactly as
        # given (one leading '/' is the URL separator) and open via SQLite URI
        # syntax so mode=ro is enforced by the database itself.
        return f"sqlite+aiosqlite:///file:{path.lstrip('/') if not path.startswith('//') else path[1:]}?mode=ro&uri=true"

    raise ToolError(f"Unsupported tool connection provider: {provider}")


@lru_cache(maxsize=DATABASE_MAX_LRU_CACHE_SIZE)
def get_external_sql_engine(
    connection_string: str, provider: str, statement_timeout_ms: int
) -> AsyncEngine:
    """A cached read-only engine for one external database.

    NullPool: recall-time tool calls are low-frequency, and holding no idle
    connections to someone's production database is the polite default.
    """
    url = _normalize_url(connection_string, provider, read_only=True)

    connect_args: dict = {}
    if provider == "postgres":
        connect_args["server_settings"] = {
            "default_transaction_read_only": "on",
            "statement_timeout": str(int(statement_timeout_ms)),
        }

    return create_async_engine(url, poolclass=NullPool, connect_args=connect_args)


@lru_cache(maxsize=DATABASE_MAX_LRU_CACHE_SIZE)
def get_external_sql_write_engine(
    connection_string: str, provider: str, statement_timeout_ms: int
) -> AsyncEngine:
    """A cached WRITABLE engine — used only by the approval-gated write path.

    Nothing else may import this: reads go through the read-only engine, and
    a write proposal only reaches this engine after the deployment gate, the
    per-connection ``allow_writes`` flag, and (for apply) an explicit
    approval call.
    """
    url = _normalize_url(connection_string, provider, read_only=False)

    connect_args: dict = {}
    if provider == "postgres":
        connect_args["server_settings"] = {
            "statement_timeout": str(int(statement_timeout_ms)),
        }

    return create_async_engine(url, poolclass=NullPool, connect_args=connect_args)


async def introspect_schema(
    engine: AsyncEngine,
    allowed_tables: list[str] | None = None,
    max_tables: int = 50,
) -> dict:
    """Table → columns/PK/FK map for the prompt, capped and filterable.

    Uses SQLAlchemy inspection so SQLite and Postgres share one code path.
    """

    def _inspect(sync_connection):
        inspector = inspect(sync_connection)
        table_names = inspector.get_table_names()
        if allowed_tables:
            allowed = set(allowed_tables)
            table_names = [name for name in table_names if name in allowed]
        table_names = sorted(table_names)[: int(max_tables)]

        schema: dict = {}
        for table_name in table_names:
            columns = [
                {"name": column["name"], "type": str(column["type"])}
                for column in inspector.get_columns(table_name)
            ]
            pk_constraint = inspector.get_pk_constraint(table_name) or {}
            primary_key = pk_constraint.get("constrained_columns") or []
            foreign_keys = [
                {
                    "column": ", ".join(fk.get("constrained_columns") or []),
                    "ref_table": fk.get("referred_table"),
                    "ref_column": ", ".join(fk.get("referred_columns") or []),
                }
                for fk in inspector.get_foreign_keys(table_name)
            ]
            schema[table_name] = {
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
            }
        return schema

    async with engine.connect() as connection:
        return await connection.run_sync(_inspect)
