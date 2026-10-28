"""Plugs the Slack connector into a cognee server (cognee >= 1.6.5).

Everything under ``cognee_host/`` may import cognee; it is installed with the
``cognee`` extra (``pip install "cognee-connector-slack[cognee]"``). The package
``__init__`` never imports this subpackage, and nothing here runs at import: a
host calls :func:`register` and mounts what :func:`routers` lists.

Moved from cognee.modules.integrations.slack.__init__ (cognee dev, 2026-10-27),
whose import-time ``use_integration(SlackIntegration())`` became the explicit
:func:`register`; the two routers cognee's ``api/client.py`` mounted by hand are
what :func:`routers` returns.

The history import (``history``, ``history_client``) reports its progress through
``cognee.modules.integrations.sync_progress`` when the installed cognee has it
(newer than 1.6.6); on an older cognee it runs the same way with every report
dropped (:mod:`cognee_connector_slack.cognee_host.compat`), so :func:`register`
and :func:`routers` work on any cognee >= 1.6.5.
"""

from __future__ import annotations

# Imported eagerly on purpose. ``routers`` is both the submodule that holds the
# route factories and the function a host calls, and importing a submodule binds
# it as an attribute of its package — a function of the same name defined here
# would be replaced by the module the first time anything imported
# ``cognee_connector_slack.cognee_host.routers`` (a test patching a route's
# collaborators does). Re-exporting the function after that import is what makes
# ``from cognee_connector_slack.cognee_host import routers`` the function,
# whatever was imported first.
from cognee_connector_slack.cognee_host.routers import routers


def register() -> None:
    """Register this provider with cognee's integration registry (cognee >= 1.6.5)."""
    from cognee.modules.integrations.registry import use_integration

    from cognee_connector_slack.cognee_host.adapter import SlackIntegration

    use_integration(SlackIntegration())


__all__ = ["register", "routers"]
