"""Per-plugin connection status, aggregated from three sources.

The status endpoint answers "which plugins are connected?" from, in order
of authority:

* **identity** — the caller's agent sub-users that carry a ``plugin``
  entry in their principal configuration (written by the provision
  endpoint, with the plugin's self-declared label and connection type).
  Connected means an API key currently exists for that agent (revocation
  deletes the row, so existence == active); activity joins the key's
  throttled ``last_used_at`` with the agent's session records.
* **sessions-legacy** — pre-migration installs that still authenticate
  with the shared tenant key. Inferred from the legacy session-id prefixes
  in :data:`BUILTIN_PLUGINS`, only for plugins with no provisioned identity.
* **registry** — the agent-connection registry
  (:func:`list_persisted_agent_connections`), bucketed onto the plugin key
  each connection names in ``metadata.plugin_key`` (any well-formed key, so
  a new integration needs no server change, SDK-646), or for connections
  that predate that field onto the frozen connection-type map, so e.g. MCP
  clients registered before first traffic show up. Connections stored in
  child-agent blobs (agent-writable) are pinned to the agent's
  email-derived identity rather than trusted.

Same key from several sources merges: ``connected`` is the OR — unless the
base row is identity-sourced, whose key-existence check is authoritative —
``last_active_at`` the max, and the more authoritative row keeps
``agent_id``/``provisioned_at``/``source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import case, func, or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.agents.registry import AGENT_CONFIG_NAME, list_persisted_agent_connections
from cognee.modules.integrations.agent_identity import plugin_key_from_agent_email
from cognee.modules.integrations.plugins import (
    BUILTIN_PLUGINS,
    PLUGIN_LABEL_MAX_LENGTH,
    is_connection_type,
    is_plugin_key,
)
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.session_lifecycle.visibility import child_agent_emails
from cognee.modules.users.methods.get_principal_configuration import (
    get_principal_all_configuration,
)
from cognee.modules.users.models.UserApiKey import UserApiKey

SOURCE_IDENTITY = "identity"
SOURCE_SESSIONS_LEGACY = "sessions-legacy"
SOURCE_REGISTRY = "registry"

# AgentConnectionType -> plugin key for connections that carry no
# ``metadata.plugin_key`` (plugin versions from before they named
# themselves, and bare MCP/SDK clients), plus the generic "sdk" bucket,
# which lands on the "api" card alongside direct API usage. Frozen: a
# plugin that wants its own card names itself.
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
    agent_id: UUID | None = None
    provisioned_at: datetime | None = None
    last_active_at: datetime | None = None
    session_count: int = 0
    source: str | None = None
    # The plugin's self-description from its provision call, when it gave one.
    label: str | None = None
    connection_type: str | None = None


def as_utc(value: datetime | None) -> datetime | None:
    """Normalize to aware-UTC; SQLite hands back tz-naive datetimes.

    Public: the integrations router runs credential timestamps through this
    too, so every datetime in the status payload serializes with an offset.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _max_datetime(*values: datetime | None) -> datetime | None:
    present = [as_utc(value) for value in values if value is not None]
    return max(present) if present else None


def coerce_provisioned_at(value) -> datetime | None:
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


def coerce_plugin_label(value) -> str | None:
    """The plugin's display label from its ``plugin`` entry, or ``None``.

    Agent-writable like ``provisioned_at``: anything but a non-blank string
    within :data:`PLUGIN_LABEL_MAX_LENGTH` reads as absent.
    """
    if not isinstance(value, str):
        return None
    label = value.strip()
    return label if label and len(label) <= PLUGIN_LABEL_MAX_LENGTH else None


def coerce_connection_type(value) -> str | None:
    """The plugin's declared connection type from its ``plugin`` entry, or ``None``."""
    return value if is_connection_type(value) else None


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
    server-assigned deterministic email (``<plugin_key>+plugin.<parent_id>@cognee.agent``
    for any key, or the plain form of agents provisioned before the tag
    under one of the built-in keys — the same identity the provision
    endpoint resolves by), never from that configuration blob: a child
    agent can rewrite its own blob through the public configuration
    endpoint, so a config-claimed key could impersonate another plugin.
    ``connected`` means the agent currently holds an API key
    (disconnect deletes keys, so existence == active); ``last_active_at``
    is the max of the key's throttled ``last_used_at`` and the agent's
    latest session activity, so pure-recall plugins that never open
    sessions still report "last seen".
    """
    plugin_agents: dict[UUID, PluginStatusRow] = {}
    for agent_id, email in (await child_agent_emails(user_id)).items():
        plugin_key = plugin_key_from_agent_email(email, user_id)
        if plugin_key is None:
            continue
        for config in await get_principal_all_configuration(agent_id):
            if config.get("name") != AGENT_CONFIG_NAME:
                continue
            plugin_entry = (config.get("configuration") or {}).get("plugin")
            if isinstance(plugin_entry, dict) and plugin_entry:
                plugin_agents[agent_id] = PluginStatusRow(
                    key=plugin_key,
                    agent_id=agent_id,
                    provisioned_at=coerce_provisioned_at(plugin_entry.get("provisioned_at")),
                    label=coerce_plugin_label(plugin_entry.get("label")),
                    connection_type=coerce_connection_type(plugin_entry.get("connection_type")),
                    source=SOURCE_IDENTITY,
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
    for agent_id, row in plugin_agents.items():
        key_row = keys_by_agent.get(agent_id)
        session_row = sessions_by_agent.get(agent_id)
        row.connected = bool(key_row and key_row.key_count)
        row.last_active_at = _max_datetime(
            key_row.last_used_at if key_row else None,
            session_row.last_activity_at if session_row else None,
        )
        row.session_count = int(session_row.session_count) if session_row else 0
        statuses[row.key] = row
    return statuses


async def legacy_plugin_statuses(
    visible_user_ids: list[UUID], exclude_keys: set[str] | None = None
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
        for key, spec in BUILTIN_PLUGINS.items()
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
      endpoint, and by every plugin on ``POST /agents/register``) is
      trusted as-is whenever it is a well-formed key; a connection without
      one is bucketed by its connection type through the frozen map — this
      is what lets e.g. shared-key MCP clients registered before first
      traffic show up.
    * A child agent's blob is agent-writable (the public configuration
      endpoint), so every connection found there is pinned to the plugin
      key derived from that agent's server-assigned email. A blob claiming
      another plugin's key must not mark that plugin connected; child
      agents that aren't a plugin identity contribute nothing here (their
      connections still appear on ``/agents/connections``).

    Only active connections are consulted, so a row here means connected.
    """
    statuses: dict[str, PluginStatusRow] = {}

    def _add(plugin_key: str | None, connection) -> None:
        if plugin_key is None or not is_plugin_key(plugin_key):
            return
        row = statuses.setdefault(
            plugin_key, PluginStatusRow(key=plugin_key, connected=True, source=SOURCE_REGISTRY)
        )
        row.last_active_at = _max_datetime(row.last_active_at, connection.last_active_at)

    for connection in await list_persisted_agent_connections([user_id], active_only=True):
        claimed_key = (connection.metadata or {}).get("plugin_key")
        _add(
            claimed_key
            if is_plugin_key(claimed_key)
            else _CONNECTION_TYPE_PLUGIN_KEYS.get(connection.type),
            connection,
        )

    for agent_id, email in (await child_agent_emails(user_id)).items():
        email_key = plugin_key_from_agent_email(email, user_id)
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
    ``agent_id``/``provisioned_at``/``source``/``label``/``connection_type``
    — identity rows win those fields over registry echoes of the same plugin.
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
        if existing.label is None:
            existing.label = row.label
        if existing.connection_type is None:
            existing.connection_type = row.connection_type
    return merged
