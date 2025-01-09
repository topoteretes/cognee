from .create_user import create_user


async def create_default_user():
    default_user_email = "default_user@example.com"
    default_user_password = "default_password"

    user = await create_user(
        email=default_user_email,
        password=default_user_password,
        is_superuser=False,
        is_active=True,
        is_verified=True,
        auto_login=True,
    )

    return user
