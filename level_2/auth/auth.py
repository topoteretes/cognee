import os

import requests
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from starlette.status import HTTP_403_FORBIDDEN

from auth.cognito.JWTBearer import JWKS, JWTBearer, JWTAuthorizationCredentials

load_dotenv()  # Automatically load environment variables from a '.env' file.

# jwks = JWKS.parse_obj(
#     requests.get(
#         f"https://cognito-idp.{os.environ.get('eu-west-1:46372257029')}.amazonaws.com/"
#         f"{os.environ.get('eu-west-1_3VUqKzMgj')}/.well-known/jwks.json"
#     ).json()
# )
# Construct the Cognito User Pool URL using the correct syntax
region = "eu-west-1"
user_pool_id = "eu-west-1_viUyNCqKp"
cognito_url = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"

# Fetch the JWKS using the updated URL
jwks = JWKS.parse_obj(requests.get(cognito_url).json())

auth = JWTBearer(jwks)


async def get_current_user(
    credentials: JWTAuthorizationCredentials = Depends(auth)
) -> str:
    try:
        return credentials.claims["username"]
    except KeyError:
        HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Username missing")
