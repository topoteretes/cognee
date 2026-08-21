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
  show up. Connections stored in child-agent blobs (agent-writable) are
  pinned to the agent's email-derived identity rather than trusted.

Same key from several sources merges: ``connected`` is the OR — unless the
base row is identity-sourced, whose key-existence check is authoritative —
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
from cognee.modules.session_lifecycle.visibility import child_agent_emails
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
    provisioned_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    session_count: int = 0
    source: Optional[str] = None


def as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize to aware-UTC; SQLite hands back tz-naive datetimes.

    Public: the integrations router runs credential timestamps through this
    too, so every datetime in the status payload serializes with an offset.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _max_datetime(*values: Optional[datetime]) -> Optional[datetime]:
    present = [as_utc(value) for value in values if value is not None]
    return max(present) if present else None


def coerce_provisioned_at(value) -> Optional[datetime]:
    """Coerce the stored ``provisioned_at`` into an aware datetime.

    The value lives in the agent's principal-configuration blob, which the
    agent itself can rewrite through the public configuration endpoint — so
    it is untrusted input. Anything unparseable degrades to ``None`` (the
    row still surfaces) instead of blowing up DTO validation and 500ing the
    whole status page.
    """
    if isinstance(value, datetime):
        return as_utc(value)
    if isinstance(value, str):
        try:
            return as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _plugin_key_from_agent_email(email: Optional[str], parent_id: UUID) -> Optional[str]:
    """Plugin key encoded in a child agent's server-assigned email, if any.

    ``create_agent`` mints ``<plugin_key>+<parent_id>@cognee.agent``
    deterministically, so the email is the one identity signal a child
    agent cannot rewrite — unlike its principal-configuration blob.
    Returns ``None`` for agents that aren't a known plugin identity.
    """
    suffix = f"+{parent_id}@cognee.agent"
    if not (email or "").endswith(suffix):
        return None
    plugin_key = email[: -len(suffix)]
    return plugin_key if plugin_key in KNOWN_PLUGINS else None


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
    writes. The plugin *key*, however, is derived from the agent's
    server-assigned deterministic email
    (``<plugin_key>+<parent_id>@cognee.agent`` — the same identity the
    provision endpoint resolves by), never from that configuration blob: a
    child agent can rewrite its own blob through the public configuration
    endpoint, so a config-claimed key could impersonate another plugin.
    ``connected`` means the agent currently holds an API key
    (disconnect deletes keys, so existence == active); ``last_active_at``
    is the max of the key's throttled ``last_used_at`` and the agent's
    latest session activity, so pure-recall plugins that never open
    sessions still report "last seen".
    """
    plugin_agents: dict[UUID, tuple[str, Optional[datetime]]] = {}
    for agent_id, email in (await child_agent_emails(user_id)).items():
        plugin_key = _plugin_key_from_agent_email(email, user_id)
        if plugin_key is None:
            continue
        for config in await get_principal_all_configuration(agent_id):
            if config.get("name") != AGENT_CONFIG_NAME:
                continue
            plugin_entry = (config.get("configuration") or {}).get("plugin")
            if isinstance(plugin_entry, dict) and plugin_entry:
                plugin_agents[agent_id] = (
                    plugin_key,
                    coerce_provisioned_at(plugin_entry.get("provisioned_at")),
                )
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
            last_active_at=as_utc(row.last_activity_at),
            session_count=int(row.session_count),
            source=SOURCE_SESSIONS_LEGACY,
        )
        for row in rows
        if row.plugin_key is not None
    }


async def registry_plugin_statuses(user_id: UUID) -> dict[str, PluginStatusRow]:
    """Statuses from the agent-connection registry.

    Persisted connections live in principal-configuration blobs, which
    gives them two trust tiers:

    * The caller's own blob is written only by the caller, so a
      connection's claimed ``metadata.plugin_key`` (set by the provision
      endpoint) or its connection type is trusted as-is — this is what
      lets e.g. shared-key MCP clients registered before first traffic
      show up.
    * A child agent's blob is agent-writable (the public configuration
      endpoint), so every connection found there is pinned to the plugin
      key derived from that agent's server-assigned email. A blob claiming
      another plugin's key must not mark that plugin connected; child
      agents that aren't a plugin identity contribute nothing here (their
      connections still appear on ``/agents/connections``).

    Only active connections are consulted, so a row here means connected.
    """
    statuses: dict[str, PluginStatusRow] = {}

    def _add(plugin_key: Optional[str], connection) -> None:
        if plugin_key not in KNOWN_PLUGINS:
            return
        row = statuses.setdefault(
            plugin_key, PluginStatusRow(key=plugin_key, connected=True, source=SOURCE_REGISTRY)
        )
        row.last_active_at = _max_datetime(row.last_active_at, connection.last_active_at)

    for connection in await list_persisted_agent_connections([user_id], active_only=True):
        _add(
            (connection.metadata or {}).get("plugin_key")
            or _CONNECTION_TYPE_PLUGIN_KEYS.get(connection.type),
            connection,
        )

    for agent_id, email in (await child_agent_emails(user_id)).items():
        email_key = _plugin_key_from_agent_email(email, user_id)
        if email_key is None:
            continue
        for connection in await list_persisted_agent_connections([agent_id], active_only=True):
            _add(email_key, connection)

    return statuses


def merge_plugin_statuses(
    base: dict[str, PluginStatusRow], extra: dict[str, PluginStatusRow]
) -> dict[str, PluginStatusRow]:
    """Merge ``extra`` into ``base``; base rows are the more authoritative.

    Same key: ``connected`` is the OR — except when the base row is
    identity-sourced, whose key-existence check is authoritative: after
    disconnect revokes every API key, a stale registry connection (e.g.
    legacy traffic auto-typed onto the same plugin) must not flip the row
    back to connected. ``last_active_at`` is the max, ``session_count``
    the max (identity and legacy counts never coexist for a key, so max
    never double-counts), and the base row keeps its
    ``agent_id``/``provisioned_at``/``source`` — identity rows win those
    fields over registry echoes of the same plugin.
    """
    merged = dict(base)
    for key, row in extra.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        if existing.source != SOURCE_IDENTITY:
            existing.connected = existing.connected or row.connected
        existing.last_active_at = _max_datetime(existing.last_active_at, row.last_active_at)
        existing.session_count = max(existing.session_count, row.session_count)
        if existing.agent_id is None:
            existing.agent_id = row.agent_id
        if existing.provisioned_at is None:
            existing.provisioned_at = row.provisioned_at
    return merged
