import os
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import jwt
from fastapi import HTTPException
from fastapi_users.exceptions import UserNotExists

from cognee.modules.users.models import User
from cognee.modules.users.authentication.jwks.jwks_jwt_strategy import JWKSJWTStrategy


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("COGNEE_JWKS_URL", "https://mock-idp.example.com/.well-known/jwks.json")
    monkeypatch.setenv("COGNEE_JWKS_AUDIENCE", "cognee_api")
    monkeypatch.setenv("COGNEE_JWKS_ISSUER", "https://mock-idp.example.com/")
    monkeypatch.setenv("COGNEE_JWKS_AUTO_PROVISION", "True")


@pytest.fixture
def strategy(mock_env):
    return JWKSJWTStrategy()


@pytest.fixture
def mock_user_manager():
    manager = AsyncMock()
    
    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        is_active=True,
    )
    
    async def get_by_email(email):
        if email == "test@example.com":
            return user
        raise UserNotExists()

    manager.get_by_email = AsyncMock(side_effect=get_by_email)
    return manager


@pytest.fixture
def mock_jwk_client():
    with patch("jwt.PyJWKClient") as mock_client_class:
        client_instance = MagicMock()
        mock_client_class.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_jwt():
    with patch("jwt.get_unverified_header") as mock_header, \
         patch("jwt.decode") as mock_decode:
        yield mock_header, mock_decode


@pytest.mark.asyncio
async def test_valid_token_existing_user(strategy, mock_user_manager, mock_jwt, mock_jwk_client):
    mock_header, mock_decode = mock_jwt
    
    mock_header.return_value = {"kid": "key1"}
    
    mock_key = MagicMock()
    mock_key.key = "mock_public_key"
    strategy.jwk_client.get_signing_key_from_jwt = MagicMock(return_value=mock_key)
    
    mock_decode.return_value = {"email": "test@example.com", "iss": "https://mock-idp.example.com/", "aud": "cognee_api"}

    user = await strategy.read_token("mock_token", mock_user_manager)
    
    assert user is not None
    assert user.email == "test@example.com"
    mock_decode.assert_called_once_with(
        "mock_token",
        "mock_public_key",
        algorithms=["RS256", "ES256"],
        audience="cognee_api",
        issuer="https://mock-idp.example.com/",
        options={"verify_aud": True, "verify_iss": True}
    )


@pytest.mark.asyncio
async def test_invalid_signature_or_claims(strategy, mock_user_manager, mock_jwt):
    mock_header, mock_decode = mock_jwt
    
    mock_header.return_value = {"kid": "key1"}
    
    mock_key = MagicMock()
    mock_key.key = "mock_public_key"
    strategy.jwk_client.get_signing_key_from_jwt = MagicMock(return_value=mock_key)
    
    # Simulate an expired token or invalid signature exception
    mock_decode.side_effect = jwt.ExpiredSignatureError("Signature has expired")

    user = await strategy.read_token("mock_token", mock_user_manager)
    
    # fastapi-users strategy must return None on invalid token
    assert user is None


@pytest.mark.asyncio
@patch("cognee.modules.users.authentication.jwks.jwks_jwt_strategy.create_user")
async def test_auto_provisioning(mock_create_user, strategy, mock_user_manager, mock_jwt):
    mock_header, mock_decode = mock_jwt
    
    mock_header.return_value = {"kid": "key1"}
    
    mock_key = MagicMock()
    mock_key.key = "mock_public_key"
    strategy.jwk_client.get_signing_key_from_jwt = MagicMock(return_value=mock_key)
    
    # Return a sub that doesn't exist to trigger auto-provisioning
    mock_decode.return_value = {"sub": "new_user@external.idp"}
    
    new_user = User(
        id=uuid.uuid4(),
        email="new_user@external.idp",
        is_active=True,
    )
    mock_create_user.return_value = new_user

    user = await strategy.read_token("mock_token", mock_user_manager)
    
    assert user is not None
    assert user.email == "new_user@external.idp"
    
    # Ensure create_user was called
    mock_create_user.assert_called_once()
    args, kwargs = mock_create_user.call_args
    assert kwargs["email"] == "new_user@external.idp"
    assert "password" in kwargs  # Auto-generated password
    assert kwargs["is_active"] is True
    assert kwargs["is_verified"] is True


@pytest.mark.asyncio
async def test_unreachable_jwks(strategy, mock_user_manager, mock_jwt):
    mock_header, mock_decode = mock_jwt
    
    mock_header.return_value = {"kid": "key1"}
    
    # Simulate a network error when trying to fetch the JWKS
    strategy.jwk_client.get_signing_key_from_jwt = MagicMock(
        side_effect=jwt.PyJWKClientConnectionError("Failed to fetch JWKS")
    )

    with pytest.raises(HTTPException) as excinfo:
        await strategy.read_token("mock_token", mock_user_manager)
    
    assert excinfo.value.status_code == 503
    assert "JWKS endpoint unreachable" in excinfo.value.detail


@pytest.mark.asyncio
async def test_unrecognized_kid(strategy, mock_user_manager, mock_jwt):
    mock_header, mock_decode = mock_jwt
    
    mock_header.return_value = {"kid": "unknown_key"}
    
    # Simulate the key not being found in the JWKS
    strategy.jwk_client.get_signing_key_from_jwt = MagicMock(
        side_effect=jwt.PyJWKClientError("Unable to find a signing key that matches")
    )

    user = await strategy.read_token("mock_token", mock_user_manager)
    
    # Should safely return None (401) rather than crash
    assert user is None


@pytest.mark.asyncio
async def test_missing_user_identifier(strategy, mock_user_manager, mock_jwt):
    mock_header, mock_decode = mock_jwt
    
    mock_header.return_value = {"kid": "key1"}
    
    mock_key = MagicMock()
    mock_key.key = "mock_public_key"
    strategy.jwk_client.get_signing_key_from_jwt = MagicMock(return_value=mock_key)
    
    # Payload missing both 'email' and 'sub'
    mock_decode.return_value = {"iss": "https://mock-idp.example.com/", "aud": "cognee_api"}

    user = await strategy.read_token("mock_token", mock_user_manager)
    
    assert user is None
