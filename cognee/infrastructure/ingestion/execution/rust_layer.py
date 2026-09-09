import os
from typing import Callable, Any, List, Dict
from cognee.shared.logging_utils import get_logger

logger = get_logger("rust_layer")

# Flag to manually disable Rust layer if needed via environment variables
RUST_LAYER_ENABLED = os.getenv("COGNEE_RUST_CHUNKER_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)

try:
    if RUST_LAYER_ENABLED:
        import cognee_rust
    else:
        cognee_rust = None
except ImportError:
    cognee_rust = None
    logger.debug(
        "cognee_rust is not installed or failed to import. Falling back to Python implementation."
    )


def is_rust_available() -> bool:
    """Return whether the native Rust extension is available and enabled."""
    return cognee_rust is not None


def chunk_by_paragraph_rust(
    data: str,
    max_chunk_size: int,
    batch_paragraphs: bool = True,
    token_counter: Callable[[str], int] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk the input text by paragraph using the high-performance Rust execution layer.

    If Rust is not available, raises a RuntimeError.
    """
    if not is_rust_available():
        raise RuntimeError("Rust execution layer is not available.")

    # Delegate to PyO3 function
    return cognee_rust.chunk_by_paragraph_rust(
        data,
        max_chunk_size,
        batch_paragraphs,
        token_counter,
    )
