"""Confirms Slack actually self-registers when the real app is imported.

Every other test in this package builds its own bare FastAPI app + a fake
integration, which proves the router's generic logic works but never proves
the wiring that gets a REAL provider registered at process startup — that
only happens because something imports cognee.modules.integrations.slack
(today, transitively, via cognee.api.v1.slack.routers -> handle_slack_command
-> ... -> the slack package's own __init__.py side effect). This test
imports the real app object (not its lifespan — no DB/migrations run just
from importing cognee.api.client) and checks the registry directly, so a
future refactor that accidentally drops that import chain fails a test
instead of only failing silently at runtime.
"""


def test_slack_registers_itself_when_the_real_app_is_imported():
    from cognee.api.client import app  # noqa: F401 - import side effect is the point
    from cognee.modules.integrations.registry import supported_integrations

    assert "slack" in supported_integrations
