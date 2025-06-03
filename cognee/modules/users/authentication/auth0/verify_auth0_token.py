import json
from jose import jwt
from jose.exceptions import JWTError
from urllib.request import urlopen

from cognee.modules.users.exceptions.exceptions import (
    UnauthenticatedException,
    UnauthorizedException,
)
from .auth0_client import oauth

from .auth0_config import get_auth0_config

auth0_config = get_auth0_config()


class VerifyAuth0Token:
    def __init__(self):
        # This gets the JWKS from a given URL and does processing so you can
        # use any of the keys available.
        jwks_url = urlopen(f"https://{auth0_config.auth0_domain}/.well-known/jwks.json")
        self.jwks = json.loads(jwks_url.read())

    def verify(self, token: str):
        try:
            unverified_header = jwt.get_unverified_header(token)
        except JWTError:
            raise UnauthenticatedException(detail="Invalid header")

        rsa_key = None
        for key in self.jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = key
                break

        if not rsa_key:
            raise UnauthorizedException(status_code=401, detail="Invalid token")

        try:
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=auth0_config.auth0_algorithms,
                audience=auth0_config.auth0_api_audience,
                issuer=f"https://{auth0_config.auth0_domain}/"
            )

            email = payload["email"]

            return email
        except JWTError as e:
            raise UnauthorizedException(detail=f"Token decode error: {str(e)}")
