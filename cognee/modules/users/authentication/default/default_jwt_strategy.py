from ..password_bound_jwt_strategy import PasswordBoundJWTStrategy


class DefaultJWTStrategy(PasswordBoundJWTStrategy):
    """Cookie-session tokens, revoked by a password change (see PasswordBoundJWTStrategy)."""
