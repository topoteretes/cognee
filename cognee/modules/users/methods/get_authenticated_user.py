from ..get_fastapi_users import get_fastapi_users


fastapi_users = get_fastapi_users()

get_authenticated_user = fastapi_users.current_user(active=True)

# from types import SimpleNamespace

# from ..get_fastapi_users import get_fastapi_users
# from fastapi import HTTPException, Security
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# import os
# import jwt

# from uuid import UUID

# fastapi_users = get_fastapi_users()

# # Allows Swagger to understand authorization type and allow single sign on for the Swagger docs to test backend
# bearer_scheme = HTTPBearer(scheme_name="BearerAuth", description="Paste **Bearer &lt;JWT&gt;**")


# async def get_authenticated_user(
#     creds: HTTPAuthorizationCredentials = Security(bearer_scheme),
# ) -> SimpleNamespace:
#     """
#     Extract and validate the JWT presented in the Authorization header.
#     """
#     if creds is None:  # header missing
#         raise HTTPException(status_code=401, detail="Not authenticated")

#     if creds.scheme.lower() != "bearer":  # shouldn't happen extra guard
#         raise HTTPException(status_code=401, detail="Invalid authentication scheme")

#     token = creds.credentials
#     try:
#         payload = jwt.decode(
#             token, os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret"), algorithms=["HS256"]
#         )

#         auth_data = SimpleNamespace(id=UUID(payload["user_id"]))
#         return auth_data

#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.InvalidTokenError:
#         raise HTTPException(status_code=401, detail="Invalid token")
