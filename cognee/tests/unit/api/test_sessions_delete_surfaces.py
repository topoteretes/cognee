"""Wiring of the per-session delete on its HTTP and CLI surfaces.

The delete itself runs over real stores in
cognee/tests/integration/infrastructure/session/test_session_delete.py; here
the HTTP route and the CLI subcommand are checked to call it for the caller's
own user, and to map "nothing to delete" to 404 / a non-zero exit.
"""

import argparse
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.sessions.routers import get_sessions_router as router_module
from cognee.cli.commands.sessions_command import SessionsCommand
from cognee.cli.exceptions import CliCommandException

delete_module = importlib.import_module("cognee.modules.session_lifecycle.delete_session")


def _client(user_id) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_sessions_router(), prefix="/api/v1/sessions")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )
    return TestClient(app)


def test_http_delete_removes_the_callers_session():
    user_id = uuid4()
    with patch.object(delete_module, "delete_session", new=AsyncMock(return_value=True)) as delete:
        response = _client(user_id).delete("/api/v1/sessions/claude-code-1")

    assert response.status_code == 200
    assert response.json() == {"session_id": "claude-code-1", "deleted": True}
    delete.assert_awaited_once_with(user_id, "claude-code-1")


def test_http_delete_of_an_unknown_session_is_404():
    with patch.object(delete_module, "delete_session", new=AsyncMock(return_value=False)):
        response = _client(uuid4()).delete("/api/v1/sessions/nope")

    assert response.status_code == 404


def _cli_args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    SessionsCommand().configure_parser(parser)
    return parser.parse_args(list(argv))


def _cli_user():
    user = MagicMock()
    user.id = uuid4()
    return user


@pytest.mark.parametrize("deleted", [True, False])
def test_cli_delete_calls_the_sdk_for_the_resolved_user(deleted):
    user = _cli_user()
    with (
        patch("cognee.cli.user_resolution.resolve_cli_user", new=AsyncMock(return_value=user)),
        patch("cognee.api.v1.session.delete_session", new=AsyncMock(return_value=deleted)) as sdk,
    ):
        args = _cli_args("delete", "s1", "--force")
        if deleted:
            SessionsCommand().execute(args)
        else:
            with pytest.raises(CliCommandException):
                SessionsCommand().execute(args)

    sdk.assert_awaited_once_with(session_id="s1", user=user)


def test_cli_delete_asks_first_and_stops_on_no():
    with (
        patch("cognee.cli.commands.sessions_command.fmt.confirm", return_value=False) as confirm,
        patch("cognee.api.v1.session.delete_session", new=AsyncMock()) as sdk,
    ):
        SessionsCommand().execute(_cli_args("delete", "s1"))

    confirm.assert_called_once()
    sdk.assert_not_awaited()
