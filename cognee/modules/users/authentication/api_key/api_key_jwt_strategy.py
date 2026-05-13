from fastapi_users.authentication import JWTStrategy

from cognee.modules.users.get_user_manager import UserManager


class ApiKeyJWTStrategy(JWTStrategy):
    def __init__(self):
        pass

    async def read_token(self, token: str, user_manager: UserManager):
        user = await user_manager.get_by_token(token=token)

        return user
