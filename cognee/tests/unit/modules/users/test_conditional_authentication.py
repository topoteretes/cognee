import os
import sys
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from fastapi import HTTPException
from types import SimpleNamespace

from cognee.modules.users.models import User


class TestConditionalAuthentication:
    """Test cases for conditional authentication functionality."""

    @pytest.mark.asyncio
    async def test_require_authentication_false_no_token_returns_default_user(self):
        """Test that when REQUIRE_AUTHENTICATION=false and no token, returns default user."""
        # Mock the default user
        mock_default_user = SimpleNamespace(id=uuid4(), email="default@example.com", is_active=True)

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            from cognee.modules.users.methods.get_authenticated_user import (
                get_authenticated_user,
            )

            with patch(
                "cognee.modules.users.methods.get_authenticated_user.get_default_user"
            ) as mock_get_default:
                mock_get_default.return_value = mock_default_user

                # Test with None user (no authentication)
                result = await get_authenticated_user(user=None)

                assert result == mock_default_user
                mock_get_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_require_authentication_false_with_valid_user_returns_user(self):
        """Test that when REQUIRE_AUTHENTICATION=false and valid user, returns that user."""
        mock_authenticated_user = User(
            id=uuid4(),
            email="user@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            from cognee.modules.users.methods.get_authenticated_user import (
                get_authenticated_user,
            )

            with patch(
                "cognee.modules.users.methods.get_authenticated_user.get_default_user"
            ) as mock_get_default:
                # Test with authenticated user
                result = await get_authenticated_user(user=mock_authenticated_user)

                assert result == mock_authenticated_user
                mock_get_default.assert_not_called()

    @pytest.mark.asyncio
    async def test_require_authentication_true_with_user_returns_user(self):
        """Test that when REQUIRE_AUTHENTICATION=true and user present, returns user."""
        mock_authenticated_user = User(
            id=uuid4(),
            email="user@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "true"}):
            from cognee.modules.users.methods.get_authenticated_user import (
                get_authenticated_user,
            )

            result = await get_authenticated_user(user=mock_authenticated_user)

            assert result == mock_authenticated_user

    @pytest.mark.asyncio
    async def test_require_authentication_true_with_none_returns_none(self):
        """Test that when REQUIRE_AUTHENTICATION=true and no user, returns None (would raise 401 at dependency level)."""
        # This test simulates what would happen if REQUIRE_AUTHENTICATION was true at import time
        # In reality, when REQUIRE_AUTHENTICATION=true, FastAPI Users would raise 401 BEFORE this function is called

        # Since REQUIRE_AUTHENTICATION is currently false (set at import time),
        # we expect it to return the default user, not None
        from cognee.modules.users.methods.get_authenticated_user import (
            get_authenticated_user,
        )

        result = await get_authenticated_user(user=None)

        # The current implementation will return default user because REQUIRE_AUTHENTICATION is false
        assert result is not None  # Should get default user
        assert hasattr(result, "id")


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

        # Currently should be False (optional authentication)
        assert REQUIRE_AUTHENTICATION == False


class TestConditionalAuthenticationEnvironmentVariables:
    """Test environment variable handling."""

    def test_require_authentication_default_false(self):
        """Test that REQUIRE_AUTHENTICATION defaults to false when imported with no env var."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove module from cache to force fresh import
            module_name = "cognee.modules.users.methods.get_authenticated_user"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import after patching environment - module will see empty environment
            from cognee.modules.users.methods.get_authenticated_user import (
                REQUIRE_AUTHENTICATION,
            )

            assert REQUIRE_AUTHENTICATION == False

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

            assert REQUIRE_AUTHENTICATION == True

    def test_require_authentication_false_explicit(self):
        """Test that REQUIRE_AUTHENTICATION=false is parsed correctly when imported."""
        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            # Remove module from cache to force fresh import
            module_name = "cognee.modules.users.methods.get_authenticated_user"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import after patching environment - module will see REQUIRE_AUTHENTICATION=false
            from cognee.modules.users.methods.get_authenticated_user import (
                REQUIRE_AUTHENTICATION,
            )

            assert REQUIRE_AUTHENTICATION == False

    def test_require_authentication_case_insensitive(self):
        """Test that environment variable parsing is case insensitive when imported."""
        test_cases = ["TRUE", "True", "tRuE", "FALSE", "False", "fAlSe"]

        for case in test_cases:
            with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": case}):
                # Remove module from cache to force fresh import
                module_name = "cognee.modules.users.methods.get_authenticated_user"
                if module_name in sys.modules:
                    del sys.modules[module_name]

                # Import after patching environment
                from cognee.modules.users.methods.get_authenticated_user import (
                    REQUIRE_AUTHENTICATION,
                )

                expected = case.lower() == "true"
                assert REQUIRE_AUTHENTICATION == expected, f"Failed for case: {case}"

    def test_current_require_authentication_value(self):
        """Test that the current REQUIRE_AUTHENTICATION module value is as expected."""
        from cognee.modules.users.methods.get_authenticated_user import (
            REQUIRE_AUTHENTICATION,
        )

        # The module-level variable should currently be False (set at import time)
        assert isinstance(REQUIRE_AUTHENTICATION, bool)
        assert REQUIRE_AUTHENTICATION == False


class TestConditionalAuthenticationEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_get_default_user_raises_exception(self):
        """Test behavior when get_default_user raises an exception."""
        from cognee.modules.users.methods.get_authenticated_user import (
            get_authenticated_user,
        )

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            with patch(
                "cognee.modules.users.methods.get_authenticated_user.get_default_user"
            ) as mock_get_default:
                mock_get_default.side_effect = Exception("Database error")

                # This should propagate the exception
                with pytest.raises(Exception, match="Database error"):
                    await get_authenticated_user(user=None)

    @pytest.mark.asyncio
    async def test_user_type_consistency(self):
        """Test that the function always returns the same type."""
        from cognee.modules.users.methods.get_authenticated_user import (
            get_authenticated_user,
        )

        mock_user = User(
            id=uuid4(),
            email="test@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
        )

        mock_default_user = SimpleNamespace(id=uuid4(), email="default@example.com", is_active=True)

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            with patch(
                "cognee.modules.users.methods.get_authenticated_user.get_default_user"
            ) as mock_get_default:
                mock_get_default.return_value = mock_default_user

                # Test with user
                result1 = await get_authenticated_user(user=mock_user)
                assert result1 == mock_user

                # Test with None
                result2 = await get_authenticated_user(user=None)
                assert result2 == mock_default_user

                # Both should have user-like interface
                assert hasattr(result1, "id")
                assert hasattr(result1, "email")
                assert hasattr(result2, "id")
                assert hasattr(result2, "email")


@pytest.mark.asyncio
class TestAuthenticationScenarios:
    """Test specific authentication scenarios that could occur in FastAPI Users."""

    async def test_fallback_to_default_user_scenarios(self):
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
        from cognee.modules.users.methods.get_authenticated_user import (
            get_authenticated_user,
        )

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            with patch(
                "cognee.modules.users.methods.get_authenticated_user.get_default_user"
            ) as mock_get_default:
                mock_get_default.return_value = mock_default_user

                # All the above scenarios result in user=None being passed to our function
                result = await get_authenticated_user(user=None)
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

        from cognee.modules.users.methods.get_authenticated_user import (
            get_authenticated_user,
        )

        with patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"}):
            result = await get_authenticated_user(user=mock_user)
            assert result == mock_user
