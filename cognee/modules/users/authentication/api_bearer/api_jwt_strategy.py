from ..password_bound_jwt_strategy import PasswordBoundJWTStrategy


class APIJWTStrategy(PasswordBoundJWTStrategy):
    """Bearer tokens, revoked by a password change (see PasswordBoundJWTStrategy).

    Must match DefaultJWTStrategy: /auth/login issues one token through the cookie
    strategy and returns it as the Bearer access_token too.
    """
