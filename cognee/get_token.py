import jwt
import datetime

from cognee.modules.users.authentication.get_auth_secret import (
    JWT_SECRET_ENV_VAR,
    get_auth_secret,
)


def create_jwt(user_id: str, tenant_id: str, roles: list[str]):
    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "roles": roles,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),  # 1 hour expiry
    }
    return jwt.encode(payload, get_auth_secret(JWT_SECRET_ENV_VAR), algorithm="HS256")


if __name__ == "__main__":
    # Example token generation
    token = create_jwt(
        "6763554c-91bd-432c-aba8-d42cd72ed659", "4523544d-82bd-432c-aca7-d42cd72ed651", ["admin"]
    )
    print(token)
