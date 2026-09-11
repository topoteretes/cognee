"""Confirms real providers actually self-register when the real app is imported.

Every other test in this package builds its own bare FastAPI app + a fake
integration, which proves the router's generic logic works but never proves
the wiring that gets a REAL provider registered at process startup — that
only happens because something imports the provider package (Slack
transitively via cognee.api.v1.slack.routers -> handle_slack_command -> ...
-> the slack package's own __init__.py side effect; GitHub and Linear only
via the bare ``import cognee.modules.integrations.<provider>  # noqa: F401``
lines in cognee/api/client.py — exactly the kind of "unused" import a
cleanup can silently drop). This test imports the real app object (not its
lifespan — no DB/migrations run just from importing cognee.api.client) and
checks the registry directly, so a future refactor that accidentally drops
any of those import chains fails a test instead of only failing silently at
runtime.
"""


def test_real_providers_register_themselves_when_the_real_app_is_imported():
    from cognee.api.client import app  # noqa: F401 - import side effect is the point
    from cognee.modules.integrations.registry import supported_integrations

    assert {"slack", "github", "linear"} <= set(supported_integrations)
