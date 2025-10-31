import os
import sys
import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from types import SimpleNamespace
import importlib


from cognee.modules.users.models import User


gau_mod = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")


class TestConditionalAuthentication:
    """Test cases for conditional authentication functionality."""

    @pytest.mark.asyncio
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    async def test_require_authentication_false_no_token_returns_default_user(
        self, mock_get_default
    ):
        """Test that when REQUIRE_AUTHENTICATION=false and no token, returns default user."""
        # Mock the default user
        mock_default_user = SimpleNamespace(id=uuid4(), email="default@example.com", is_active=True)
        mock_get_default.return_value = mock_default_user

        # Use gau_mod.get_authenticated_user instead

        # Test with None user (no authentication)
        result = await gau_mod.get_authenticated_user(user=None)

        assert result == mock_default_user
        mock_get_default.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    async def test_require_authentication_false_with_valid_user_returns_user(
        self, mock_get_default
    ):
        """Test that when REQUIRE_AUTHENTICATION=false and valid user, returns that user."""
        mock_authenticated_user = User(
            id=uuid4(),
            email="user@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        # Use gau_mod.get_authenticated_user instead

        # Test with authenticated user
        result = await gau_mod.get_authenticated_user(user=mock_authenticated_user)

        assert result == mock_authenticated_user
        mock_get_default.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    async def test_require_authentication_true_with_user_returns_user(self, mock_get_default):
        """Test that when REQUIRE_AUTHENTICATION=true and user present, returns user."""
        mock_authenticated_user = User(
            id=uuid4(),
            email="user@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        # Use gau_mod.get_authenticated_user instead

        result = await gau_mod.get_authenticated_user(user=mock_authenticated_user)

        assert result == mock_authenticated_user


class TestConditionalAuthenticationIntegration:
    """Integration tests that test the full authentication flow."""

    @pytest.mark.asyncio
    async def test_fastapi_users_dependency_creation(self):
        """Test that FastAPI Users dependency can be created correctly."""
        from cognee.modules.users.get_fastapi_users import get_fastapi_users

        fastapi_users = get_fastapi_users()

        # Test that we can create optional dependency
        optional_dependency = fastapi_users.current_user(optional=True, active=True)
        assert callable(optional_dependency)

        # Test that we can create required dependency
        required_dependency = fastapi_users.current_user(active=True)  # optional=False by default
        assert callable(required_dependency)

    @pytest.mark.asyncio
    async def test_conditional_authentication_function_exists(self):
        """Test that the conditional authentication function can be imported and used."""
        from cognee.modules.users.methods.get_authenticated_user import (
            get_authenticated_user,
            REQUIRE_AUTHENTICATION,
        )

        # Should be callable
        assert callable(get_authenticated_user)

        # REQUIRE_AUTHENTICATION should be a boolean
        assert isinstance(REQUIRE_AUTHENTICATION, bool)


class TestConditionalAuthenticationEnvironmentVariables:
    """Test environment variable handling."""

    def test_require_authentication_true(self):
        """Test that REQUIRE_AUTHENTICATION=true is parsed correctly when imported."""
        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "true"}):
            # Remove module from cache to force fresh import
            module_name = "cognee.modules.users.methods.get_authenticated_user"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import after patching environment - module will see REQUIRE_AUTHENTICATION=true
            from cognee.modules.users.methods.get_authenticated_user import (
                REQUIRE_AUTHENTICATION,
            )

            assert REQUIRE_AUTHENTICATION


class TestConditionalAuthenticationEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    async def test_get_default_user_raises_exception(self, mock_get_default):
        """Test behavior when get_default_user raises an exception."""
        mock_get_default.side_effect = Exception("Database error")

        # This should propagate the exception
        with pytest.raises(Exception, match="Database error"):
            await gau_mod.get_authenticated_user(user=None)

    @pytest.mark.asyncio
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    async def test_user_type_consistency(self, mock_get_default):
        """Test that the function always returns the same type."""
        mock_user = User(
            id=uuid4(),
            email="test@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        mock_default_user = SimpleNamespace(id=uuid4(), email="default@example.com", is_active=True)
        mock_get_default.return_value = mock_default_user

        # Test with user
        result1 = await gau_mod.get_authenticated_user(user=mock_user)
        assert result1 == mock_user

        # Test with None
        result2 = await gau_mod.get_authenticated_user(user=None)
        assert result2 == mock_default_user

        # Both should have user-like interface
        assert hasattr(result1, "id")
        assert hasattr(result1, "email")
        assert result1.id == mock_user.id
        assert result1.email == mock_user.email
        assert hasattr(result2, "id")
        assert hasattr(result2, "email")
        assert result2.id == mock_default_user.id
        assert result2.email == mock_default_user.email


@pytest.mark.asyncio
class TestAuthenticationScenarios:
    """Test specific authentication scenarios that could occur in FastAPI Users."""

    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    async def test_fallback_to_default_user_scenarios(self, mock_get_default):
        """
        Test fallback to default user for all scenarios where FastAPI Users returns None:
        - No JWT/Cookie present
        - Invalid JWT/Cookie
        - Valid JWT but user doesn't exist in database
        - Valid JWT but user is inactive (active=True requirement)

        All these scenarios result in FastAPI Users returning None when optional=True,
        which should trigger fallback to default user.
        """
        mock_default_user = SimpleNamespace(id=uuid4(), email="default@example.com")
        mock_get_default.return_value = mock_default_user

        # All the above scenarios result in user=None being passed to our function
        result = await gau_mod.get_authenticated_user(user=None)
        assert result == mock_default_user
        mock_get_default.assert_called_once()

    async def test_scenario_valid_active_user(self):
        """Scenario: Valid JWT and user exists and is active → returns the user."""
        mock_user = User(
            id=uuid4(),
            email="active@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        # Use gau_mod.get_authenticated_user instead

        result = await gau_mod.get_authenticated_user(user=mock_user)
        assert result == mock_user
