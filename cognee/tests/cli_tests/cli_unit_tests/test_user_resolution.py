"""Unit tests for cognee.cli.user_resolution."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID, uuid4

import pytest

from cognee.cli.user_resolution import (
    resolve_cli_user,
    scoped_session_id,
    _get_default_user_with_recovery,
)


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

    def test_valid_uuid_unknown_user_raises(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        mock_get_user = AsyncMock(side_effect=Exception("not found"))
        mock_get_default = AsyncMock()

        with patch("cognee.modules.users.methods.get_user", mock_get_user):
            with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
                with pytest.raises(ValueError, match="does not exist"):
                    asyncio.run(resolve_cli_user(uid))
                mock_get_default.assert_not_awaited()


class TestGetDefaultUserWithRecovery:
    """Tests for _get_default_user_with_recovery (issue #3267).

    On a fresh Postgres database, get_default_user() raises
    DatabaseNotCreatedError because `principals` doesn't exist yet.
    The recovery helper catches that, runs migrations, and retries.
    """

    def test_no_recovery_when_get_default_user_succeeds(self):
        mock_user = MagicMock()
        mock_get_default = AsyncMock(return_value=mock_user)
        mock_run_migrations = AsyncMock(return_value=[])

        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            with patch(
                "cognee.modules.migrations.startup.run_migrations",
                mock_run_migrations,
            ):
                result = asyncio.run(_get_default_user_with_recovery())

        assert result is mock_user
        mock_get_default.assert_awaited_once()
        mock_run_migrations.assert_not_awaited()

    def test_recovery_on_database_not_created_error(self):
        from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError

        mock_user = MagicMock()
        mock_get_default = AsyncMock(side_effect=[DatabaseNotCreatedError(), mock_user])
        mock_run_migrations = AsyncMock(return_value=[])

        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            with patch(
                "cognee.modules.migrations.startup.run_migrations",
                mock_run_migrations,
            ):
                result = asyncio.run(_get_default_user_with_recovery())

        assert result is mock_user
        assert mock_get_default.await_count == 2
        mock_run_migrations.assert_awaited_once()

    def test_recovery_creates_db_when_migrations_fail(self):
        from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError

        mock_user = MagicMock()
        mock_get_default = AsyncMock(side_effect=[DatabaseNotCreatedError(), mock_user])
        mock_run_migrations = AsyncMock(side_effect=[Exception("alembic fails"), []])
        mock_engine = MagicMock()
        mock_engine.create_database = AsyncMock(return_value=None)
        mock_get_engine = MagicMock(return_value=mock_engine)

        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            with patch(
                "cognee.modules.migrations.startup.run_migrations",
                mock_run_migrations,
            ):
                with patch(
                    "cognee.infrastructure.databases.relational.get_relational_engine",
                    mock_get_engine,
                ):
                    result = asyncio.run(_get_default_user_with_recovery())

        assert result is mock_user
        assert mock_run_migrations.await_count == 2
        mock_engine.create_database.assert_awaited_once()

    def test_recovery_propagates_if_retry_fails(self):
        from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError

        mock_get_default = AsyncMock(
            side_effect=[DatabaseNotCreatedError(), RuntimeError("still broken")]
        )
        mock_run_migrations = AsyncMock(return_value=[])

        with patch("cognee.modules.users.methods.get_default_user", mock_get_default):
            with patch(
                "cognee.modules.migrations.startup.run_migrations",
                mock_run_migrations,
            ):
                with pytest.raises(RuntimeError, match="still broken"):
                    asyncio.run(_get_default_user_with_recovery())
