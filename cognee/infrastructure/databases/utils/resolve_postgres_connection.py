"""Shared Postgres connection resolution for the vector and graph stores.

Both the pgvector vector adapter and the postgres graph adapter need to build a
Postgres connection from either an explicit URL or a set of discrete credential
parts, with a fall back to the unified ``DB_*`` relational block when the store
does not configure its own credentials. This module centralizes that precedence
so the two call sites stay consistent and use ``URL.create`` for escaping.
"""

from typing import Optional, Union

from sqlalchemy import URL
from sqlalchemy.engine import make_url

from cognee.shared.logging_utils import get_logger

logger = get_logger("PostgresConnectionResolver")


def resolve_postgres_url(
    *,
    store_url: str,
    store_host: str,
    store_port: Union[str, int, None],
    store_username: str,
    store_password: str,
    store_name: str,
    shared_host: str,
    shared_port: Union[str, int, None],
    shared_username: str,
    shared_password: str,
    shared_name: str,
    missing_error: str,
    driver: str = "postgresql+asyncpg",
) -> URL:
    """Resolve a Postgres connection for a single store under a unified precedence.

    A store-level URL, if set, is authoritative and returned as-is. Otherwise the
    discrete connection parts are resolved as one atomic group: if the store
    supplies its own core credentials, the store's parts (including its port) are
    used; if it does not, the entire shared ``DB_*`` block — host, port, username,
    password, and name together — is used. Ports are never inferred from
    placeholder values; a store's port is used only when the store's own
    credential group is chosen, and the shared port is used only when the shared
    group is chosen. The URL is assembled with ``sqlalchemy.URL.create``, which
    percent-escapes special characters in usernames and passwords (``#``, ``@``,
    ``:``).

    Parameters:
    -----------

        - store_url (str): The store's own connection URL (e.g. ``VECTOR_DB_URL``
          / ``GRAPH_DATABASE_URL``). When set, it wins over everything.
        - store_host / store_port / store_username / store_password / store_name:
          The store's discrete credential parts.
        - shared_host / shared_port / shared_username / shared_password /
          shared_name: The unified ``DB_*`` relational block, used as a group when
          the store does not configure its own credentials.
        - missing_error (str): The ``EnvironmentError`` message raised when neither
          group is complete. Owned by the caller so each store keeps its wording.
        - driver (str): The SQLAlchemy driver name. Defaults to
          ``postgresql+asyncpg``.

    Returns:
    --------

        - URL: A SQLAlchemy ``URL`` built from the resolved connection info.
    """
    # A fully-formed URL is authoritative and bypasses part resolution entirely.
    if store_url:
        return make_url(store_url)

    # The credential parts are resolved as one group. The presence of the store's
    # core credentials (host/username/password/name) decides which group wins —
    # port is intentionally excluded from this trigger so placeholder port
    # defaults are never compared, and travels with whichever group is chosen.
    store_has_own = bool(store_host and store_username and store_password and store_name)

    if store_has_own:
        host, port, username, password, name = (
            store_host,
            store_port,
            store_username,
            store_password,
            store_name,
        )
    else:
        host, port, username, password, name = (
            shared_host,
            shared_port,
            shared_username,
            shared_password,
            shared_name,
        )

    if not (host and username and password and name):
        raise EnvironmentError(missing_error)

    return URL.create(
        driver,
        username=username,
        password=password,
        host=host,
        port=int(port) if port else None,
        database=name,
    )


def warn_unknown_db_env_vars(model_extra: Optional[dict], prefixes: tuple) -> None:
    """Warn about unknown env vars captured under the given DB prefixes.

    ``BaseSettings`` is configured with ``extra="allow"``, so a typo'd database
    env var (e.g. ``DB_HSOT``) is silently absorbed into ``model_extra`` instead
    of being rejected. This helper emits a targeted warning for any extra key
    whose name matches one of the supplied database prefixes, so typos surface
    without breaking unrelated env vars users keep in their ``.env``.

    Parameters:
    -----------

        - model_extra (dict | None): The pydantic ``model_extra`` mapping of
          captured-but-undeclared fields. ``None`` is treated as empty.
        - prefixes (tuple): The database env-var prefixes to scope the scan to
          (e.g. ``("db_", "vector_db_", "graph_database_")``).
    """
    if not model_extra:
        return

    normalized_prefixes = tuple(prefix.lower() for prefix in prefixes)

    for key in model_extra:
        if key.lower().startswith(normalized_prefixes):
            logger.warning(
                "Unknown database environment variable '%s' is not a recognized "
                "configuration field and will be ignored. Check for typos.",
                key,
            )
