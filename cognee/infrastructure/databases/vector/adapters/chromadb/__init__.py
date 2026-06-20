"""ChromaDB vector adapter (community registration)."""

from cognee.infrastructure.databases.vector.use_vector_adapter import use_vector_adapter

try:
    from .ChromaDBAdapter import ChromaDBAdapter

    use_vector_adapter("chromadb", ChromaDBAdapter)
except ImportError:
    ChromaDBAdapter = None  # type: ignore[misc, assignment]

__all__ = ["ChromaDBAdapter"]
