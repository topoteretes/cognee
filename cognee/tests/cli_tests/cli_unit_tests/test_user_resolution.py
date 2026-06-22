"""Unit tests for cognee.cli.user_resolution."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID, uuid4

import pytest

from cognee.cli.user_resolution import resolve_cli_user, scoped_session_id


class TestScopedSessionId:
    def test_default_session(self):
        uid = UUID("550e8400-e29b-41d4-a716-446655440000")
        assert scoped_session_id(uid) == "550e8400-e29b-41d4-a716-446655440000:default"

    def test_custom_session(self):
        uid = UUID("550e8400-e29b-41d4-a716-446655440000")
        assert (
            scoped_session_id(uid, "chat-1")
            == "550e8400-e29b-41d4-a716-446655440000:chat-1"
        )

    def test_none_session_uses_default(self):
        uid = uuid4()
        result = scoped_session_id(uid, None)
        assert result.endswith(":default")

    def test_different_users_different_sessions(self):
        u1 = uuid4()
        u2 = uuid4()
        assert scoped_session_id(u1) != scoped_session_id(u2)


class TestResolveCliUser:
    @pytest.fixture(autouse=True)
    def mock_database_setup(self):
        with patch(
            "cognee.modules.engine.operations.setup.setup", new_callable=AsyncMock
        ) as mock_setup:
            yield mock_setup

    def test_none_returns_default_user(self, mock_database_setup):
        mock_user = MagicMock()
        mock_get_default = AsyncMock(return_value=mock_user)
        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            result = asyncio.run(resolve_cli_user(None))
            assert result is mock_user
            mock_database_setup.assert_awaited_once()

    def test_empty_string_returns_default_user(self, mock_database_setup):
        mock_user = MagicMock()
        mock_get_default = AsyncMock(return_value=mock_user)
        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            result = asyncio.run(resolve_cli_user(""))
            assert result is mock_user
            mock_database_setup.assert_awaited_once()

    def test_invalid_uuid_raises_value_error(self, mock_database_setup):
        with pytest.raises(ValueError, match="not a valid UUID"):
            asyncio.run(resolve_cli_user("not-a-uuid"))
        mock_database_setup.assert_not_awaited()

    def test_valid_uuid_existing_user(self, mock_database_setup):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        mock_user = MagicMock()
        mock_get_user = AsyncMock(return_value=mock_user)
        with patch("cognee.modules.users.methods.get_user", mock_get_user):
            result = asyncio.run(resolve_cli_user(uid))
            assert result is mock_user
            mock_database_setup.assert_awaited_once()

    def test_database_setup_happens_before_default_user_lookup(
        self, mock_database_setup
    ):
        calls = []

        async def setup_side_effect():
            calls.append("setup")

        async def get_default_side_effect():
            calls.append("get_default_user")
            return MagicMock()

        mock_database_setup.side_effect = setup_side_effect
        mock_get_default = AsyncMock(side_effect=get_default_side_effect)

        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            asyncio.run(resolve_cli_user(None))

        assert calls == ["setup", "get_default_user"]

    def test_valid_uuid_unknown_user_warns_and_falls_back(self, mock_database_setup):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        default_user = MagicMock()
        mock_get_user = AsyncMock(side_effect=Exception("not found"))
        mock_get_default = AsyncMock(return_value=default_user)

        with patch("cognee.modules.users.methods.get_user", mock_get_user):
            with patch(
                "cognee.modules.users.methods.get_default_user", mock_get_default
            ):
                with patch("cognee.cli.echo.warning") as mock_warn:
                    result = asyncio.run(resolve_cli_user(uid))
                    assert result is default_user
                    mock_database_setup.assert_awaited_once()
                    mock_warn.assert_called_once()
                    assert "falling back" in mock_warn.call_args[0][0].lower()
