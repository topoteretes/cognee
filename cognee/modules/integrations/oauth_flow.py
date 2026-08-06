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

The state format is ``{field_1}:...:{field_n}:{expires}:{hmac}`` — plain, not
encoded, since none of the fields need to survive arbitrary transport beyond
a URL query parameter. Every provider today signs exactly one field (the
initiating user id), which is why ``make_state``/``validate_state`` default
to that shape — a caller that needs to bind more than one id to the same
install (e.g. a deployment scoping the callback to both a workspace/org id
and the initiating user) passes more fields and reads them back with a
matching ``field_count``.
"""

import hmac
import hashlib
import time
from typing import Optional, Union
from uuid import UUID

# Long enough to pick a workspace/account and click through the provider's
# consent screen; short enough that a leaked state is stale before it's
# useful. Providers needing a different window pass their own ttl_seconds.
DEFAULT_STATE_TTL_SECONDS = 60 * 10


def sign_state_payload(payload: str, *, signing_secret: str) -> str:
    """HMAC-SHA256 ``payload`` under ``signing_secret``, hex-encoded."""
    return hmac.new(signing_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_state(
    *fields: Union[str, UUID],
    signing_secret: str,
    ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS,
) -> str:
    """Mint the CSRF state binding an OAuth install to its initiating fields.

    ``fields`` are signed in the order given and must be read back with the
    same count via ``validate_state(..., field_count=len(fields))``. One
    field (the initiating user id) is today's only real usage across every
    provider.
    """
    if not fields:
        raise ValueError("make_state requires at least one field")

    expires = int(time.time()) + ttl_seconds
    payload = ":".join([*(str(field) for field in fields), str(expires)])
    return f"{payload}:{sign_state_payload(payload, signing_secret=signing_secret)}"


def validate_state(
    state: str,
    *,
    signing_secret: str,
    field_count: int = 1,
) -> Optional[Union[UUID, tuple]]:
    """Return the signed fields for a valid, unexpired state; ``None`` otherwise.

    Verifies the HMAC before reading any field, so a forged or tampered
    state never influences behavior — not even error messages.

    With the default ``field_count=1`` (every provider today), returns the
    single field parsed as a ``UUID`` — the original single-field contract,
    unchanged. With ``field_count`` > 1, returns a tuple of the raw string
    fields in the order ``make_state`` received them; the caller knows what
    each position means and converts as needed.
    """
    parts = (state or "").split(":")
    if len(parts) != field_count + 2:
        return None

    *field_strs, expires_str, signature = parts
    payload = ":".join([*field_strs, expires_str])

    if not hmac.compare_digest(
        sign_state_payload(payload, signing_secret=signing_secret), signature
    ):
        return None

    try:
        if int(expires_str) < time.time():
            return None
    except ValueError:
        return None

    if field_count == 1:
        try:
            return UUID(field_strs[0])
        except ValueError:
            return None

    return tuple(field_strs)
