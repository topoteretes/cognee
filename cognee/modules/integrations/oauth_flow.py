"""Provider-agnostic OAuth2 ``state`` signing and validation.

Every OAuth2 install flow needs a CSRF token that also binds the incoming
callback to whichever cognee user started it — the callback itself is
necessarily unauthenticated (the browser arrives from the provider's site
without a session header), so a valid, unexpired, unforged state is the
*only* thing tying an inbound ``code`` back to a user. That requirement is
identical across every provider; only the signing secret differs. This
module is the one place that logic lives — extracted out of the Slack
adapter (see :mod:`cognee.modules.integrations.slack.oauth`, which now just
supplies its own signing secret to the functions here) so a second provider
never has to re-derive or copy this HMAC scheme.

The state format is ``{user_id}:{expires}:{hmac}`` — plain, not encoded,
since none of the three fields need to survive arbitrary transport beyond a
URL query parameter.
"""

import hmac
import hashlib
import time
from typing import Optional
from uuid import UUID

# Long enough to pick a workspace/account and click through the provider's
# consent screen; short enough that a leaked state is stale before it's
# useful. Providers needing a different window pass their own ttl_seconds.
DEFAULT_STATE_TTL_SECONDS = 60 * 10


def sign_state_payload(payload: str, *, signing_secret: str) -> str:
    """HMAC-SHA256 ``payload`` under ``signing_secret``, hex-encoded."""
    return hmac.new(signing_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_state(
    user_id: UUID,
    *,
    signing_secret: str,
    ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS,
) -> str:
    """Mint the CSRF state binding an OAuth install to its initiating user."""
    expires = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{expires}"
    return f"{payload}:{sign_state_payload(payload, signing_secret=signing_secret)}"


def validate_state(state: str, *, signing_secret: str) -> Optional[UUID]:
    """Return the ``user_id`` for a valid, unexpired state; ``None`` otherwise.

    Verifies the HMAC before reading any field, so a forged or tampered
    state never influences behavior — not even error messages.
    """
    parts = (state or "").split(":")
    if len(parts) != 3:
        return None
    user_id_str, expires_str, signature = parts

    payload = f"{user_id_str}:{expires_str}"
    if not hmac.compare_digest(
        sign_state_payload(payload, signing_secret=signing_secret), signature
    ):
        return None

    try:
        if int(expires_str) < time.time():
            return None
        return UUID(user_id_str)
    except ValueError:
        return None
