from fastapi import HTTPException
from fastapi_users.schemas import BaseUserCreate
from fastapi_users.exceptions import UserNotExists
from fastapi_users.authentication import JWTStrategy

from cognee.modules.users.get_user_manager import UserManager
from cognee.modules.users.tenants.methods import create_tenant
from cognee.modules.users.authentication.auth0.verify_auth0_token import VerifyAuth0Token


token_verification = VerifyAuth0Token()


class Auth0JWTStrategy(JWTStrategy):
    async def read_token(self, token: str, user_manager: UserManager):
        email = token_verification.verify(token)

        if not email:
            raise HTTPException(status_code=400, detail="Missing email in token")

        try:
            user = await user_manager.get_by_email(user_email=email)

            return user
        except UserNotExists:
            # Auto-provision user

            new_user = BaseUserCreate(
                email=email,
                password="NOT IMPORTANT FOR AUTH0"
            )

            user = await user_manager.create(new_user)
            await create_tenant(tenant_name="My organization", user_id=user.id)

        return user
