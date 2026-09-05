import os
import pytest
from unittest.mock import patch
from cognee.tasks.chunks.chunk_by_paragraph import chunk_by_paragraph
from cognee.infrastructure.ingestion.execution import is_rust_available

@pytest.mark.skipif(not is_rust_available(), reason="Rust extension not installed")
def test_rust_chunker_equivalence():
    """Test that Rust-based chunk_by_paragraph produces the exact same results as Python-based."""
    test_text = (
        "Cognee is an open-source framework for building knowledge graphs. "
        "It helps developers build semantic search and retrieval systems.\n\n"
        "This is the second paragraph of the text. It contains some sentence endings. "
        "Will the Rust chunker split it exactly the same way as the Python chunker?\n\n"
        "Let's see if the zero-copy implementation matches character for character!"
    )
    
    # Get Rust chunker results
    rust_chunks = list(chunk_by_paragraph(test_text, max_chunk_size=15, batch_paragraphs=True))
    
    # Get Python chunker results by mocking is_rust_available to return False
    with patch("cognee.infrastructure.ingestion.execution.is_rust_available", return_value=False):
        py_chunks = list(chunk_by_paragraph(test_text, max_chunk_size=15, batch_paragraphs=True))
        
    assert len(rust_chunks) == len(py_chunks), "Number of chunks do not match"
    
    for r_chunk, py_chunk in zip(rust_chunks, py_chunks):
        assert r_chunk["text"] == py_chunk["text"], f"Texts do not match: {r_chunk['text']!r} vs {py_chunk['text']!r}"
        assert r_chunk["chunk_size"] == py_chunk["chunk_size"], "Chunk sizes do not match"
        assert r_chunk["chunk_index"] == py_chunk["chunk_index"], "Chunk indexes do not match"
        assert r_chunk["cut_type"] == py_chunk["cut_type"], "Cut types do not match"
        assert len(r_chunk["paragraph_ids"]) == len(py_chunk["paragraph_ids"]), "Number of paragraph IDs do not match"


@pytest.mark.skipif(not is_rust_available(), reason="Rust extension not installed")
@patch.dict(os.environ, {"COGNEE_RUST_CHUNKER_ENABLED": "false"})
def test_rust_chunker_disabled_fallback():
    """Verify that setting COGNEE_RUST_CHUNKER_ENABLED=false falls back to Python."""
    # When disabled, importing is_rust_available should return False or be disabled
    # Force reload of execution module config
    import importlib
    import cognee.infrastructure.ingestion.execution.rust_layer as rust_layer
    
    importlib.reload(rust_layer)
    assert not rust_layer.is_rust_available()
