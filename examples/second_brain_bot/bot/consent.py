"""Per-user opt-in / opt-out, keyed by canonical user.

Default is opt-in: a personal second brain captures the owner's own notes from
first contact, which keeps the demo frictionless. Set ``default_opt_in=False``
for a privacy-forward deployment that requires an explicit /optin first.

In-memory by default, matching the #3608 design note on small pluggable stores.
"""

from __future__ import annotations


class ConsentStore:
    def __init__(self, default_opt_in: bool = True) -> None:
        self._default = default_opt_in
        # canonical_user_id -> capturing? (True = capture, False = paused)
        self._flags: dict[str, bool] = {}

    def is_allowed(self, canonical_user_id: str) -> bool:
        return self._flags.get(canonical_user_id, self._default)

    def opt_in(self, canonical_user_id: str) -> None:
        self._flags[canonical_user_id] = True

    def opt_out(self, canonical_user_id: str) -> None:
        self._flags[canonical_user_id] = False

    def reset(self, canonical_user_id: str) -> None:
        """Forget any stored preference (back to default). Used by /forget me."""
        self._flags.pop(canonical_user_id, None)
