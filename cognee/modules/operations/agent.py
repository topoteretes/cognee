"""Which agent the current operation runs as (SDK-383).

Agents are child ``User`` rows (``parent_user_id`` set) holding their own API
key; provisioned plugins are agents named after their plugin key. A request
authenticated with an agent's key already records the agent as ``user_id``,
but nothing marks it as an agent, ties it to its owner, or names the plugin.

A ContextVar carries that identity so every telemetry event emitted for the
request carries it — including the many emitters that pass ``"sdk"`` or a
bare user id rather than the ``User`` model. Set once at authentication:

- per HTTP request, by ``get_authenticated_user``
- per WebSocket connection, by ``get_authenticated_websocket_user``

ContextVars propagate into ``asyncio`` tasks created afterwards, so background
work started by the request inherits the agent.

This module must stay import-light (stdlib plus the import-free plugin
registry) so ``send_telemetry`` can read it without cycles.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from cognee.modules.integrations.plugins import KNOWN_PLUGINS

logger = logging.getLogger(__name__)

# Agents created under a name that is not a known plugin key. Their name is
# free text typed by a user, so telemetry never carries it.
CUSTOM_AGENT_PLUGIN = "custom"


@dataclass(frozen=True)
class AgentIdentity:
    """An agent principal, reduced to the fields telemetry may carry."""

    agent_id: str
    owner_user_id: str
    plugin: str

    def telemetry_properties(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "agent_owner_id": self.owner_user_id,
            "agent_plugin": self.plugin,
        }


_operation_agent: ContextVar[AgentIdentity | None] = ContextVar(
    "cognee_operation_agent", default=None
)
_REQUEST_STATE_ATTR = "operation_agent"


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


def agent_identity_from_user(user) -> AgentIdentity | None:
    """The agent identity of ``user``, or ``None`` when it is not an agent.

    ``user`` may be anything a telemetry caller holds — a ``User`` model, a
    UUID, the ``"sdk"`` sentinel — and only a model with ``parent_user_id``
    set is an agent. Attribute access is guarded because an expired or
    detached ORM instance raises on it, and attribution must never break
    the operation it describes.
    """
    try:
        parent_user_id = getattr(user, "parent_user_id", None)
        if parent_user_id is None:
            return None
        agent_id = user.id
        email = getattr(user, "email", None)
    except Exception:
        logger.debug("Ignoring exception in agent_identity_from_user", exc_info=True)
        return None
    plugin = plugin_key_from_agent_email(email, parent_user_id) or CUSTOM_AGENT_PLUGIN
    return AgentIdentity(agent_id=str(agent_id), owner_user_id=str(parent_user_id), plugin=plugin)


def get_operation_agent() -> AgentIdentity | None:
    """The agent the current work runs as; ``None`` for a non-agent caller."""
    return _operation_agent.get()


def request_operation_agent(request) -> AgentIdentity | None:
    """The agent authentication stored on ``request.state``, if any.

    ``@app.middleware("http")`` runs the rest of the app in a child task, so
    the ContextVar binding made during authentication never flows back out
    to it. ``request.state`` lives on the ASGI scope, which both sides share,
    so the outer exception middleware reads the agent from here instead.
    """
    return getattr(getattr(request, "state", None), _REQUEST_STATE_ATTR, None)


def set_operation_agent(agent: AgentIdentity | None, request=None) -> None:
    """Bind the agent for the current context and everything spawned from it.

    Pass the HTTP ``request`` to also store it on ``request.state``, where
    code running outside the handler's context can read it (see
    :func:`request_operation_agent`).
    """
    _operation_agent.set(agent)
    if request is not None:
        setattr(request.state, _REQUEST_STATE_ATTR, agent)


@contextmanager
def operation_agent_scope(agent: AgentIdentity | None) -> Iterator[None]:
    """Temporarily override the agent, restoring the previous one on exit."""
    token = _operation_agent.set(agent)
    try:
        yield
    finally:
        _operation_agent.reset(token)
