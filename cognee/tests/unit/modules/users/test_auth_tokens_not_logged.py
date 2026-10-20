"""Password-reset and verification tokens must never reach the logs.

`on_after_forgot_password` and `on_after_request_verify` used to log the raw
token at INFO. `POST /auth/forgot-password` takes only an email and needs no
login, so anyone who can read the logs (a log aggregator, `docker logs`, a
shared host) could request a reset for any account and take it over with the
token from the log line. The hooks still record that the event happened.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from cognee.modules.users.get_user_manager import UserManager

TOKEN = "reset-token-must-not-be-logged-7f3a"


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["on_after_forgot_password", "on_after_request_verify"])
async def test_auth_token_is_not_logged(hook, caplog):
    manager = UserManager(MagicMock())
    user = SimpleNamespace(id=uuid4())

    with caplog.at_level(logging.DEBUG, logger="cognee.modules.users.get_user_manager"):
        await getattr(manager, hook)(user, TOKEN)

    assert TOKEN not in caplog.text
    assert str(user.id) in caplog.text
