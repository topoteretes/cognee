"""Opt-in / opt-out consent, the privacy gate in front of ingestion.

The adapter never remembers a message from a user whose consent is off. The
default is a deliberate policy choice rather than a hardcoded value:

* In a **1:1 / direct** context, using the bot *is* the opt-in, so the default
  is allow.
* In a **group / channel** context, one member cannot consent on everyone's
  behalf, so the default is deny until each user explicitly opts in.

The store is a tiny pluggable interface with an in-memory default, so the core
has no hard backend dependency. A production bot swaps in a Redis/SQL-backed
store implementing the same three methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConsentStore(Protocol):
    """Per-user opt-in/opt-out flags.

    ``is_set`` distinguishes "user made an explicit choice" from "no choice yet"
    so the adapter can fall back to the context default only for the latter.
    """

    def is_set(self, user: str) -> bool:
        """True if this user has explicitly opted in or out."""
        ...

    def get(self, user: str) -> bool:
        """The user's explicit choice. Only meaningful when ``is_set`` is True."""
        ...

    def set(self, user: str, on: bool) -> None:
        """Record an explicit opt-in (``on=True``) or opt-out (``on=False``)."""
        ...


class InMemoryConsentStore:
    """Process-local :class:`ConsentStore`. The zero-config default.

    Fine for a single-process bot and for tests; not shared across workers and
    not persisted across restarts. Swap for a durable store in production.
    """

    def __init__(self) -> None:
        self._flags: dict[str, bool] = {}

    def is_set(self, user: str) -> bool:
        return user in self._flags

    def get(self, user: str) -> bool:
        return self._flags[user]

    def set(self, user: str, on: bool) -> None:
        self._flags[user] = on
