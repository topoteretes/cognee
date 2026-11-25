"""Recognise an agent from its ``User`` model (SDK-383).

Agents are child ``User`` rows (``parent_user_id`` set) holding their own API
key; provisioned plugins are agents named after their plugin key. An event
from an agent already records the agent as ``user_id``, but a bare id says
nothing about it being an agent, who owns it, or which plugin it is. Those
are read off the model here, so telemetry can carry them.

Only the model can be read: telling an agent's id from a human's needs a
database lookup, so emitters pass the ``User`` they hold, not ``user.id``.

Import-light (stdlib plus the import-free plugin registry) so
``send_telemetry`` can use it without cycles.
"""

import logging
from uuid import UUID

from cognee.modules.integrations.plugins import KNOWN_PLUGINS

logger = logging.getLogger(__name__)

# Agents created under a name that is not a known plugin key. Their name is
# free text typed by a user, so telemetry never carries it.
CUSTOM_AGENT_PLUGIN = "custom"


def plugin_key_from_agent_email(email: str | None, parent_id: UUID | str) -> str | None:
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


def agent_telemetry_properties(user) -> dict:
    """``is_agent`` / ``agent_owner_id`` / ``agent_plugin`` for ``user``.

    ``user`` may be anything a telemetry caller holds — a ``User`` model, a
    UUID, the ``"sdk"`` sentinel — and only a model with ``parent_user_id``
    set is an agent; anything else reads as not one. Attribute access is
    guarded because an expired or detached ORM instance raises on it, and
    attribution must never break the operation it describes.
    """
    not_agent = {"is_agent": False, "agent_owner_id": None, "agent_plugin": None}
    try:
        parent_user_id = getattr(user, "parent_user_id", None)
        if parent_user_id is None:
            return not_agent
        email = getattr(user, "email", None)
    except Exception:
        logger.debug("Ignoring exception in agent_telemetry_properties", exc_info=True)
        return not_agent
    return {
        "is_agent": True,
        "agent_owner_id": str(parent_user_id),
        "agent_plugin": plugin_key_from_agent_email(email, parent_user_id) or CUSTOM_AGENT_PLUGIN,
    }
