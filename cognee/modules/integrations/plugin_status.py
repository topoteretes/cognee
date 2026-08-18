"""Per-plugin connection status, aggregated from three sources.

The status endpoint answers "which plugins are connected?" from, in order
of authority:

* **identity** — the caller's agent sub-users that carry a ``plugin_key``
  in their principal configuration (written by the provision endpoint).
  Connected means an API key currently exists for that agent (revocation
  deletes the row, so existence == active); activity joins the key's
  throttled ``last_used_at`` with the agent's session records.
* **sessions-legacy** — pre-migration installs that still authenticate
  with the shared tenant key. Inferred from the legacy session-id prefixes
  in :data:`KNOWN_PLUGINS`, only for plugins with no provisioned identity.
* **registry** — the agent-connection registry
  (:func:`list_persisted_agent_connections`), bucketed onto plugin keys by
  connection type, so e.g. MCP clients registered before first traffic
  show up.

Same key from several sources merges: ``connected`` is the OR,
``last_active_at`` the max, and the more authoritative row keeps
``agent_id``/``provisioned_at``/``source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import case, func, or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.agents.registry import AGENT_CONFIG_NAME, list_persisted_agent_connections
from cognee.modules.integrations.plugins import KNOWN_PLUGINS
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.session_lifecycle.visibility import child_agent_user_ids
from cognee.modules.users.methods.get_principal_configuration import (
    get_principal_all_configuration,
)
from cognee.modules.users.models.UserApiKey import UserApiKey

SOURCE_IDENTITY = "identity"
SOURCE_SESSIONS_LEGACY = "sessions-legacy"
SOURCE_REGISTRY = "registry"

# AgentConnectionType -> KNOWN_PLUGINS key (reverse of the provision
# endpoint's plugin -> connection-type map, plus the generic "sdk" bucket,
# which lands on the "api" card alongside direct API usage).
_CONNECTION_TYPE_PLUGIN_KEYS: dict[str, str] = {
    "claude_code": "claude-code",
    "opencode": "opencode",
    "mcp": "mcp",
    "api": "api",
    "sdk": "api",
}


@dataclass
class PluginStatusRow:
    """One plugin's aggregated connection state (display fields only)."""

    key: str
    connected: bool = False
    agent_id: Optional[UUID] = None
    provisioned_at: Optional[str] = None
    last_active_at: Optional[datetime] = None
    session_count: int = 0
    source: Optional[str] = None


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize to aware-UTC; SQLite hands back tz-naive datetimes."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _max_datetime(*values: Optional[datetime]) -> Optional[datetime]:
    present = [_as_utc(value) for value in values if value is not None]
    return max(present) if present else None


def _escape_like_prefix(prefix: str) -> str:
    r"""Build a LIKE pattern matching ids that start with ``prefix`` literally.

    ``_`` and ``%`` are LIKE wildcards, so an unescaped ``cc_%`` pattern
    would also match ``ccx...`` ids. Escaped with ``\`` — pass
    ``escape="\\"`` alongside.
    """
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


async def identity_plugin_statuses(user_id: UUID) -> dict[str, PluginStatusRow]:
    """Statuses of plugins provisioned with their own agent identity.

    A child agent counts as a plugin identity iff its principal
    configuration carries the ``plugin`` entry the provision endpoint
    writes. ``connected`` means the agent currently holds an API key
    (disconnect deletes keys, so existence == active); ``last_active_at``
    is the max of the key's throttled ``last_used_at`` and the agent's
    latest session activity, so pure-recall plugins that never open
    sessions still report "last seen".
    """
    plugin_agents: dict[UUID, tuple[str, Optional[str]]] = {}
    for agent_id in await child_agent_user_ids(user_id):
        for config in await get_principal_all_configuration(agent_id):
            if config.get("name") != AGENT_CONFIG_NAME:
                continue
            plugin_entry = (config.get("configuration") or {}).get("plugin") or {}
            plugin_key = plugin_entry.get("key")
            if plugin_key in KNOWN_PLUGINS:
                plugin_agents[agent_id] = (plugin_key, plugin_entry.get("provisioned_at"))
            break

    if not plugin_agents:
        return {}

    agent_ids = list(plugin_agents)
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        key_rows = (
            await session.execute(
                select(
                    UserApiKey.user_id,
                    func.count().label("key_count"),
                    func.max(UserApiKey.last_used_at).label("last_used_at"),
                )
                .where(UserApiKey.user_id.in_(agent_ids))
                .group_by(UserApiKey.user_id)
            )
        ).all()
        session_rows = (
            await session.execute(
                select(
                    SessionRecord.user_id,
                    func.count().label("session_count"),
                    func.max(SessionRecord.last_activity_at).label("last_activity_at"),
                )
                .where(SessionRecord.user_id.in_(agent_ids))
                .group_by(SessionRecord.user_id)
            )
        ).all()

    keys_by_agent = {row.user_id: row for row in key_rows}
    sessions_by_agent = {row.user_id: row for row in session_rows}

    statuses: dict[str, PluginStatusRow] = {}
    for agent_id, (plugin_key, provisioned_at) in plugin_agents.items():
        key_row = keys_by_agent.get(agent_id)
        session_row = sessions_by_agent.get(agent_id)
        statuses[plugin_key] = PluginStatusRow(
            key=plugin_key,
            connected=bool(key_row and key_row.key_count),
            agent_id=agent_id,
            provisioned_at=provisioned_at,
            last_active_at=_max_datetime(
                key_row.last_used_at if key_row else None,
                session_row.last_activity_at if session_row else None,
            ),
            session_count=int(session_row.session_count) if session_row else 0,
            source=SOURCE_IDENTITY,
        )
    return statuses


async def legacy_plugin_statuses(
    visible_user_ids: list[UUID], exclude_keys: Optional[set[str]] = None
) -> dict[str, PluginStatusRow]:
    """Prefix-inferred statuses for pre-migration shared-key installs.

    One grouped query over the caller's visible user ids, matching each
    plugin's legacy ``session_prefix`` with the LIKE wildcards escaped —
    a ``ccx_...`` session id must never count toward the ``cc_`` prefix.
    Plugins in ``exclude_keys`` (those with a provisioned identity) are
    skipped: once a plugin has its own agent, prefix inference for it is
    stale noise, not signal.
    """
    exclude = exclude_keys or set()
    prefixes = {
        key: spec["session_prefix"]
        for key, spec in KNOWN_PLUGINS.items()
        if spec.get("session_prefix") and key not in exclude
    }
    if not prefixes or not visible_user_ids:
        return {}

    like_terms = {
        key: SessionRecord.session_id.like(_escape_like_prefix(prefix), escape="\\")
        for key, prefix in prefixes.items()
    }
    plugin_key_case = case(*[(term, key) for key, term in like_terms.items()], else_=None)

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(
                    plugin_key_case.label("plugin_key"),
                    func.count().label("session_count"),
                    func.max(SessionRecord.last_activity_at).label("last_activity_at"),
                )
                .where(
                    SessionRecord.user_id.in_(visible_user_ids),
                    or_(*like_terms.values()),
                )
                .group_by("plugin_key")
            )
        ).all()

    return {
        row.plugin_key: PluginStatusRow(
            key=row.plugin_key,
            connected=True,
            last_active_at=_as_utc(row.last_activity_at),
            session_count=int(row.session_count),
            source=SOURCE_SESSIONS_LEGACY,
        )
        for row in rows
        if row.plugin_key is not None
    }


async def registry_plugin_statuses(user_ids: list[UUID]) -> dict[str, PluginStatusRow]:
    """Statuses from the agent-connection registry.

    Connections registered by the provision endpoint carry the plugin key
    in their metadata; everything else buckets by connection type. Only
    active connections are consulted, so a row here means connected.
    """
    statuses: dict[str, PluginStatusRow] = {}
    for connection in await list_persisted_agent_connections(user_ids, active_only=True):
        plugin_key = (connection.metadata or {}).get("plugin_key") or (
            _CONNECTION_TYPE_PLUGIN_KEYS.get(connection.type)
        )
        if plugin_key not in KNOWN_PLUGINS:
            continue
        row = statuses.setdefault(
            plugin_key, PluginStatusRow(key=plugin_key, connected=True, source=SOURCE_REGISTRY)
        )
        row.last_active_at = _max_datetime(row.last_active_at, connection.last_active_at)
    return statuses


def merge_plugin_statuses(
    base: dict[str, PluginStatusRow], extra: dict[str, PluginStatusRow]
) -> dict[str, PluginStatusRow]:
    """Merge ``extra`` into ``base``; base rows are the more authoritative.

    Same key: ``connected`` is the OR, ``last_active_at`` the max,
    ``session_count`` the max (identity and legacy counts never coexist for
    a key, so max never double-counts), and the base row keeps its
    ``agent_id``/``provisioned_at``/``source`` — identity rows win those
    fields over registry echoes of the same plugin.
    """
    merged = dict(base)
    for key, row in extra.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        existing.connected = existing.connected or row.connected
        existing.last_active_at = _max_datetime(existing.last_active_at, row.last_active_at)
        existing.session_count = max(existing.session_count, row.session_count)
        if existing.agent_id is None:
            existing.agent_id = row.agent_id
        if existing.provisioned_at is None:
            existing.provisioned_at = row.provisioned_at
    return merged
