import jwt
import os
import datetime

SECRET_KEY = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")


def create_jwt(user_id: str, tenant: str, role: str):
    payload = {
        "sub": user_id,
        "tenant": tenant,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),  # 1 hour expiry
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


# Example token generation
token = create_jwt("user_123", "tenant_456", "admin")
print(token)
