from cognee.modules.users.methods import create_user as create_user_method


async def create_user(email: str, password: str, is_superuser: bool = False):
    user = await create_user_method(
        email=email,
        password=password,
        is_superuser=is_superuser,
        is_verified=True,
    )

    return user
