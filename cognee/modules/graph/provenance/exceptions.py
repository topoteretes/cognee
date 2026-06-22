"""Typed error for unimplemented graph-native provenance capabilities."""


class UnsupportedProvenanceCapability(NotImplementedError):
    """Raised when a backend is asked for a graph-native provenance capability
    it has not implemented yet.

    Every new provenance method added in Part 0 (on ``GraphDBInterface`` and
    ``GraphVectorStoreInterface``) ships with a default body that raises this.
    The defaults are deliberately *not* ``@abstractmethod`` — existing adapters
    must keep importing and instantiating cleanly while Part 1 fills the
    capabilities in backend by backend. A backend that hasn't implemented a
    capability raises this typed error rather than silently passing, so callers
    (and tests) can detect the gap explicitly and fall back to the relational
    ledger.

    Subclasses ``NotImplementedError`` so existing ``except NotImplementedError``
    handlers keep working, while ``except UnsupportedProvenanceCapability`` lets
    new code distinguish "this backend lacks graph-native provenance" from any
    other ``NotImplementedError``.
    """

    def __init__(self, capability: str, backend: str | None = None):
        self.capability = capability
        self.backend = backend
        location = f" on backend '{backend}'" if backend else ""
        super().__init__(
            f"Graph-native provenance capability '{capability}' is not supported{location}. "
            "This capability is part of the delete/rollback graph-native plan and has not "
            "been implemented for this backend yet."
        )


__all__ = ["UnsupportedProvenanceCapability"]
