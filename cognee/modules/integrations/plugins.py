"""Registry of agent plugins that can be provisioned with their own identity.

Single source of truth for plugin keys, display labels, and the legacy
session-id prefixes (previously duplicated client-side, where connection
status was inferred from session-id prefixes plus localStorage). A plugin
here is a client that talks to cognee — a coding agent, an MCP client, or
direct API/SDK usage — not an OAuth data-source integration; those live in
:mod:`cognee.modules.integrations.registry`.

``session_prefix`` is the legacy heuristic marker some plugins prepend to
their session ids (``cc_…`` for claude-code). Plugins provisioned through
``POST /integrations/plugins/{plugin_key}/provision`` get their own agent
sub-user + API key instead, so the prefix is only needed to recognize
pre-migration installs; ``None`` means the plugin never had one.
"""

KNOWN_PLUGINS: dict[str, dict] = {
    "claude-code": {"label": "Claude Code", "session_prefix": "cc_"},
    "desktop": {"label": "Cognee Desktop", "session_prefix": None},
    "codex": {"label": "Codex", "session_prefix": "codex_"},
    "opencode": {"label": "OpenCode", "session_prefix": "opencode_"},
    "openclaw": {"label": "Openclaw", "session_prefix": None},
    "mcp": {"label": "MCP", "session_prefix": None},
    "api": {"label": "API/SDK", "session_prefix": None},
}


def get_plugin(plugin_key: str) -> dict:
    """Look up a known plugin by key.

    Raises:
        KeyError: ``plugin_key`` isn't a known plugin — the integrations
        router translates this into a 404, not a 500, since an unknown
        plugin in the URL is a client error, not a server fault.
    """
    return KNOWN_PLUGINS[plugin_key]
