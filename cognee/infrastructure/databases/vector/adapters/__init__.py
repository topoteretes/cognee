"""Optional community vector adapters bundled for registration via use_vector_adapter."""

try:
    from cognee.infrastructure.databases.vector.adapters import chromadb as _chromadb  # noqa: F401
except ImportError:
    pass
