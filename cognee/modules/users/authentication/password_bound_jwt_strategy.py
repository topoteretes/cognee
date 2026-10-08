"""JWT strategy whose tokens stop verifying once the user's password changes.

A stock fastapi-users access token carries only the user id, audience and expiry,
so it stays valid until it expires no matter what happens to the account: a
session opened on another device survives a password change. Binding each token
to the password it was issued under closes that. ``write_token`` adds a
fingerprint of the stored password hash, and ``read_token`` rejects a token whose
fingerprint no longer matches, so changing the password signs out every existing
session at once, the one that made the change included.

The fingerprint is an HMAC of the password hash keyed with the token secret, not
the hash itself, because a JWT payload is readable by whoever holds the token.
It is cheap to compute on every request, unlike re-hashing with the password
helper. Tokens issued before this binding existed have no fingerprint and are
rejected, which signs everyone out once on upgrade rather than leaving
unrevocable tokens in circulation.
"""

import hashlib
import hmac

import jwt
from fastapi_users import exceptions, models
from fastapi_users.authentication import JWTStrategy
from fastapi_users.jwt import decode_jwt, generate_jwt
from fastapi_users.manager import BaseUserManager
from pydantic import SecretStr

PASSWORD_FINGERPRINT_CLAIM = "pwd"


class PasswordBoundJWTStrategy(JWTStrategy[models.UP, models.ID]):
    def _password_fingerprint(self, hashed_password: str | None) -> str:
        """HMAC of the stored password hash, keyed with the token secret."""
        key = self.encode_key
        if isinstance(key, SecretStr):
            key = key.get_secret_value()
        message = (hashed_password or "").encode()
        return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()

    async def write_token(self, user: models.UP) -> str:
        data = {
            "sub": str(user.id),
            "aud": self.token_audience,
            PASSWORD_FINGERPRINT_CLAIM: self._password_fingerprint(user.hashed_password),
        }
        return generate_jwt(data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm)

    async def read_token(
        self, token: str | None, user_manager: BaseUserManager[models.UP, models.ID]
    ) -> models.UP | None:
        if token is None:
            return None

        try:
            data = decode_jwt(
                token, self.decode_key, self.token_audience, algorithms=[self.algorithm]
            )
        except jwt.PyJWTError:
            return None

        user_id = data.get("sub")
        fingerprint = data.get(PASSWORD_FINGERPRINT_CLAIM)
        if user_id is None or not isinstance(fingerprint, str):
            return None

        try:
            user = await user_manager.get(user_manager.parse_id(user_id))
        except (exceptions.UserNotExists, exceptions.InvalidID):
            return None

        expected = self._password_fingerprint(user.hashed_password)
        if not hmac.compare_digest(fingerprint, expected):
            return None
        return user
