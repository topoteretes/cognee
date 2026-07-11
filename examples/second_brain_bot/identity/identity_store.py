"""Cross-transport identity resolution: the ownable core of #3613.

A link table maps an external identity ``(transport, external_user)`` to one
``canonical_user_id``. First contact auto-creates a canonical user; ``/link``
points a second identity at an existing one, so two front-ends share one brain.

Canonical ids are deterministic (uuid5) from the first external identity, which
keeps tests reproducible and gives clean forget semantics: after wiping and
unlinking, messaging again yields a fresh empty brain, not the old one.

In-memory by default; swap the dict for a real table in production.
"""

from __future__ import annotations

import uuid

# Fixed namespace so canonical ids are stable across runs and processes.
_CANONICAL_NAMESPACE = uuid.UUID("6f3b9c2a-1d4e-5a6b-8c7d-9e0f1a2b3c4d")


class IdentityStore:
    def __init__(self) -> None:
        # (transport, external_user) -> canonical_user_id
        self._links: dict[tuple[str, str], str] = {}

    def resolve(self, transport: str, external_user: str) -> str:
        """Resolve an external identity to its canonical user, creating one on first contact."""
        key = (transport, external_user)
        if key not in self._links:
            self._links[key] = self._mint_canonical(transport, external_user)
        return self._links[key]

    def link(self, transport: str, external_user: str, canonical_user_id: str) -> None:
        """Point an external identity at an existing canonical user (merge front-ends)."""
        self._links[(transport, external_user)] = canonical_user_id

    def identities_for(self, canonical_user_id: str) -> list[tuple[str, str]]:
        """Every external identity currently linked to this canonical user."""
        return [key for key, value in self._links.items() if value == canonical_user_id]

    def unlink_all(self, canonical_user_id: str) -> None:
        """Drop every external identity link for a canonical user (used by forget)."""
        for key in self.identities_for(canonical_user_id):
            del self._links[key]

    @staticmethod
    def _mint_canonical(transport: str, external_user: str) -> str:
        return str(uuid.uuid5(_CANONICAL_NAMESPACE, f"{transport}:{external_user}"))
