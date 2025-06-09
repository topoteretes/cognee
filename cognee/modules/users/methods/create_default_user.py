from .create_user import create_user
from cognee.base_config import get_base_config


async def create_default_user():
    base_config = get_base_config()
    default_user_email = base_config.default_user_email
    default_user_password = base_config.default_user_password

    if not default_user_email or not default_user_password:
        raise RuntimeError(
            "Default superuser creation failed: "
            "Missing default_user_email and/or default_user_password configuration. "
            "These must be explicitly set to create the initial superuser."
        )

    user = await create_user(
        email=default_user_email,
        password=default_user_password,
        is_superuser=True,
        is_active=True,
        is_verified=True,
        auto_login=True,
    )

    return user