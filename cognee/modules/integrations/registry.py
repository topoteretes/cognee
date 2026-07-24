"""Registry of installed third-party OAuth integrations.

Mirrors the ``supported_databases``/``use_vector_adapter`` seam already used
by the graph and vector adapters: a plain module-level dict, populated by
``use_integration()`` at provider-package import time, never by this module
importing providers itself. A provider that isn't installed (its ``pip``
extra wasn't selected) simply never registers — callers get an unregistered
``KeyError`` at lookup, not an import error at startup. See each provider's
``__init__.py`` (e.g. Slack's) for the ``try/except ImportError`` guard that
makes registration optional.
"""

from cognee.modules.integrations.base import OAuthIntegration

supported_integrations: dict[str, OAuthIntegration] = {}


def use_integration(integration: OAuthIntegration) -> None:
    """Register ``integration`` under its own ``provider`` name."""
    supported_integrations[integration.provider] = integration


def get_integration(provider: str) -> OAuthIntegration:
    """Look up a registered integration by provider name.

    Raises:
        KeyError: ``provider`` isn't registered — the caller (the generic
        integrations router) translates this into a 404, not a 500, since an
        unknown provider in the URL is a client error, not a server fault.
    """
    return supported_integrations[provider]
