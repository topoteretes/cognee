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
        assert scoped_session_id(uid, "chat-1") == "550e8400-e29b-41d4-a716-446655440000:chat-1"

    def test_none_session_uses_default(self):
        uid = uuid4()
        result = scoped_session_id(uid, None)
        assert result.endswith(":default")

    def test_different_users_different_sessions(self):
        u1 = uuid4()
        u2 = uuid4()
        assert scoped_session_id(u1) != scoped_session_id(u2)


class TestResolveCliUser:
    def test_none_returns_default_user(self):
        mock_user = MagicMock()
        mock_get_default = AsyncMock(return_value=mock_user)
        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            result = asyncio.run(resolve_cli_user(None))
            assert result is mock_user

    def test_empty_string_returns_default_user(self):
        mock_user = MagicMock()
        mock_get_default = AsyncMock(return_value=mock_user)
        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            result = asyncio.run(resolve_cli_user(""))
            assert result is mock_user

    def test_invalid_uuid_raises_value_error(self):
        with pytest.raises(ValueError, match="not a valid UUID"):
            asyncio.run(resolve_cli_user("not-a-uuid"))

    def test_valid_uuid_existing_user(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        mock_user = MagicMock()
        mock_get_user = AsyncMock(return_value=mock_user)
        with patch("cognee.modules.users.methods.get_user", mock_get_user):
            result = asyncio.run(resolve_cli_user(uid))
            assert result is mock_user

    def test_valid_uuid_unknown_user_warns_and_falls_back(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        default_user = MagicMock()
        mock_get_user = AsyncMock(side_effect=Exception("not found"))
        mock_get_default = AsyncMock(return_value=default_user)

        with patch("cognee.modules.users.methods.get_user", mock_get_user):
            with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
                with patch("cognee.cli.echo.warning") as mock_warn:
                    result = asyncio.run(resolve_cli_user(uid))
                    assert result is default_user
                    mock_warn.assert_called_once()
                    assert "falling back" in mock_warn.call_args[0][0].lower()
