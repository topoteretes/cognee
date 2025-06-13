import jwt
from uuid import UUID
from fastapi_users.jwt import generate_jwt
from fastapi_users.authentication import JWTStrategy

from cognee.modules.users.models import User
from cognee.modules.users.get_user_manager import UserManager


class DefaultJWTStrategy(JWTStrategy):
    # async def read_token(self, token: str, user_manager: UserManager):
    #     payload = jwt.decode(token, self.secret, algorithms=["HS256"])

    #     user_id = UUID(payload["user_id"])

    #     return await user_manager.get(user_id)

    # async def write_token(self, user: User) -> str:
    #     # JoinLoad tenant and role information to user object
    #     data = {"user_id": str(user.id)}

    #     return generate_jwt(data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm)
    pass
