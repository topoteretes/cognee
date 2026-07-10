"""One-time-code linking flow for merging two front-ends onto one brain.

Flow:
    1. /link on transport A issues a short code tied to A's canonical user,
       with a short TTL.
    2. Entering that code on transport B redeems it once: B's external
       identity is pointed at A's canonical user. Both now share one brain.

The clock is injectable so TTL behavior is deterministic in tests. Codes are
single-use and expire.
"""

from __future__ import annotations

import secrets
import time
from typing import Callable, Optional

from .identity_store import IdentityStore


class LinkingService:
    def __init__(
        self,
        identity_store: IdentityStore,
        ttl_seconds: int = 600,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._store = identity_store
        self._ttl = ttl_seconds
        self._clock = clock or time.time
        # code -> (canonical_user_id, expires_at)
        self._codes: dict[str, tuple[str, float]] = {}

    def issue_code(self, canonical_user_id: str) -> str:
        """Issue a one-time link code for the given canonical user."""
        code = self._mint_code()
        self._codes[code] = (canonical_user_id, self._clock() + self._ttl)
        return code

    def redeem_code(self, code: str, transport: str, external_user: str) -> Optional[str]:
        """Redeem a code from transport B, linking it to the issuer's brain.

        Returns the canonical user id on success, or None if the code is
        unknown or expired.
        """
        # Minted codes are lowercase hex; normalize so a code retyped with a
        # different case (e.g. mobile autocapitalization) still redeems.
        code = code.strip().lower()
        entry = self._codes.get(code)
        if entry is None:
            return None

        canonical_user_id, expires_at = entry
        if self._clock() > expires_at:
            del self._codes[code]
            return None

        self._store.link(transport, external_user, canonical_user_id)
        del self._codes[code]  # one-time use
        return canonical_user_id

    @staticmethod
    def _mint_code() -> str:
        # Six lowercase hex chars: short enough to type, unguessable enough for a TTL window.
        return secrets.token_hex(3)
