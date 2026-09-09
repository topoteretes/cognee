import os
import uuid
import secrets
from typing import Optional

import jwt
from fastapi import HTTPException, status
from fastapi_users.authentication.strategy.base import Strategy
from fastapi_users.exceptions import UserNotExists

from cognee.modules.users.get_user_manager import UserManager
from cognee.modules.users.models import User
from cognee.modules.users.methods.create_user import create_user


class JWKSJWTStrategy(Strategy[User, uuid.UUID]):
    def __init__(self):
        self.jwks_url = os.getenv("COGNEE_JWKS_URL")
        self.audience = os.getenv("COGNEE_JWKS_AUDIENCE")
        self.issuer = os.getenv("COGNEE_JWKS_ISSUER")
        self.auto_provision = os.getenv("COGNEE_JWKS_AUTO_PROVISION", "True").lower() in ("true", "1", "yes")

        # PyJWKClient automatically fetches and caches the JWKS based on cache configuration.
        self.jwk_client = jwt.PyJWKClient(self.jwks_url)

    async def read_token(self, token: Optional[str], user_manager: UserManager) -> Optional[User]:
        if not token:
            return None

        try:
            # Fetch the matching signing key from the cached JWKS
            # PyJWKClient internally decodes the unverified header to get the 'kid' (Key ID)
            signing_key = self.jwk_client.get_signing_key_from_jwt(token)
            
            # Build dynamic options to safely handle optional claims
            decode_options = {
                "verify_aud": bool(self.audience),
                "verify_iss": bool(self.issuer),
            }

            # Verify the token signature and standard claims
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self.audience,
                issuer=self.issuer,
                options=decode_options,
            )

            # Try to resolve user by email, or fallback to sub claim if no email is provided
            user_identifier = payload.get("email") or payload.get("sub")
            
            if not user_identifier:
                return None

            try:
                user = await user_manager.get_by_email(user_identifier)
                return user
            except UserNotExists:
                if self.auto_provision:
                    # Auto-provision the missing user
                    # Generate a secure random password since the user relies on the external IdP
                    random_password = secrets.token_urlsafe(32)
                    new_user = await create_user(
                        email=user_identifier,
                        password=random_password,
                        is_active=True,
                        is_verified=True,
                    )
                    return new_user
                
                return None

        except jwt.PyJWKClientConnectionError as e:
            # 503 Service Unavailable if the JWKS endpoint itself is unreachable
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"JWKS endpoint unreachable: {str(e)}",
            )
        except (jwt.PyJWTError, jwt.PyJWKClientError):
            # Failed signature, expired token, wrong issuer/audience, missing kid, etc.
            # Return None to trigger fastapi-users default 401 Unauthorized behavior
            return None

    async def write_token(self, user: User) -> str:
        # This strategy is read-only (tokens are minted by the external IdP)
        raise NotImplementedError("JWKS JWT Strategy is read-only.")

    async def destroy_token(self, token: str, user: User) -> None:
        # External IdPs handle their own token destruction/invalidation
        pass
