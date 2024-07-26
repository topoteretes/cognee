from cognee.infrastructure.databases.relational.user_authentication.authentication_db import create_user_method



async def create_user(email: str, password: str, is_superuser: bool = False):
    output = await create_user_method(email=email, password=password, is_superuser=is_superuser)
    return output