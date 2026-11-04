"""Plugin keys: the grammar every plugin key obeys, and the frozen built-ins.

A plugin is a client that talks to cognee — a coding agent, an MCP client, or
direct API/SDK usage — not an OAuth data-source integration; those live in
:mod:`cognee.modules.integrations.registry`.

The server keeps no list of plugins that may provision an identity (SDK-646).
A plugin names itself: its key is the path of
``POST /integrations/plugins/{plugin_key}/provision`` and the
``metadata.plugin_key`` of ``POST /agents/register``, and the provision body
carries its display ``label`` and ``connection_type``. Any key that matches
:data:`PLUGIN_KEY_PATTERN` is a plugin key.

:data:`BUILTIN_PLUGINS` is frozen at the SDK-646 cut and is never extended. It
serves three legacy purposes only: the integrations page shows these seven as
cards even before they connect (labels included), ``session_prefix`` recognizes
pre-identity installs that still prepend a marker to their session ids, and an
agent created before the provision route tagged its emails (see
:func:`cognee.modules.integrations.agent_identity.plugin_key_from_agent_email`)
is a plugin identity only under one of these keys.
"""

import re

# Lowercase slug: alphanumeric runs joined by ``-``, ``.`` or ``_``, at most 64
# characters. Matches the integration slugs of cognee-integrations'
# ``inventory.yml`` (``hermes-agent``, ``openclaw-skills``, ``claude-agent-sdk``).
# ``create_agent`` lowercases a name and turns spaces into dashes, so a key in
# this grammar round-trips through the agent's email unchanged.
PLUGIN_KEY_PATTERN = re.compile(r"^[a-z0-9]+(?:[-._][a-z0-9]+)*$")
PLUGIN_KEY_MAX_LENGTH = 64

# A self-declared connection type (``claude_code``, ``hermes_agent``): one
# lowercase word, like ``KNOWN_AGENT_CONNECTION_TYPES`` in the agents module.
CONNECTION_TYPE_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")

PLUGIN_LABEL_MAX_LENGTH = 64

BUILTIN_PLUGINS: dict[str, dict] = {
    "claude-code": {"label": "Claude Code", "session_prefix": "cc_"},
    "desktop": {"label": "Cognee Desktop", "session_prefix": None},
    "codex": {"label": "Codex", "session_prefix": "codex_"},
    "opencode": {"label": "OpenCode", "session_prefix": "opencode_"},
    "openclaw": {"label": "Openclaw", "session_prefix": None},
    "mcp": {"label": "MCP", "session_prefix": None},
    "api": {"label": "API/SDK", "session_prefix": None},
}


def is_plugin_key(value) -> bool:
    """Whether ``value`` is a well-formed plugin key."""
    return (
        isinstance(value, str)
        and len(value) <= PLUGIN_KEY_MAX_LENGTH
        and PLUGIN_KEY_PATTERN.fullmatch(value) is not None
    )


def is_connection_type(value) -> bool:
    """Whether ``value`` is a well-formed self-declared connection type."""
    return isinstance(value, str) and CONNECTION_TYPE_PATTERN.fullmatch(value) is not None


def default_connection_type(plugin_key: str) -> str:
    """The connection type a plugin key implies when the plugin declares none.

    Every integration declares its key with dashes turned into underscores
    (``claude-code`` → ``claude_code``, ``hermes-agent`` → ``hermes_agent``),
    so a plugin version that predates the self-description gets the same type.
    A ``.`` becomes ``_`` too, since :data:`CONNECTION_TYPE_PATTERN` allows
    no dot; ``foo.bar`` and ``foo-bar`` therefore share a default type, which
    is harmless because status rows are bucketed by the key, never the type.
    """
    return plugin_key.replace("-", "_").replace(".", "_")
