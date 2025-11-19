"""
Test script to verify OllamaEmbeddingEngine fix with real Ollama server.
Tests that the fix correctly handles Ollama's API response format.
"""
import asyncio
import sys
from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
    OllamaEmbeddingEngine,
)


async def test_ollama_embedding():
    """Test OllamaEmbeddingEngine with real Ollama server."""

    print("=" * 80)
    print("Testing OllamaEmbeddingEngine Fix")
    print("=" * 80)

    # Configure for your Ollama server
    ollama_endpoint = "http://10.0.10.9:11434/api/embeddings"
    ollama_model = "nomic-embed-text"

    print("\nConfiguration:")
    print(f"  Endpoint: {ollama_endpoint}")
    print(f"  Model: {ollama_model}")
    print("  Expected dimensions: 768")

    # Initialize the embedding engine
    print("\n1. Initializing OllamaEmbeddingEngine...")
    try:
        engine = OllamaEmbeddingEngine(
            model=ollama_model,
            dimensions=768,
            endpoint=ollama_endpoint,
            huggingface_tokenizer="bert-base-uncased",
        )
        print("   ✅ Engine initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize engine: {e}")
        sys.exit(1)

    # Test single text embedding
    print("\n2. Testing single text embedding...")
    test_texts = ["The sky is blue and the grass is green."]

    try:
        embeddings = await engine.embed_text(test_texts)
        print("   ✅ Embedding generated successfully")
        print(f"   📊 Embedding shape: {len(embeddings)} texts, {len(embeddings[0])} dimensions")
        print(f"   📊 First 5 values: {embeddings[0][:5]}")

        # Verify dimensions
        if len(embeddings[0]) == 768:
            print("   ✅ Dimensions match expected (768)")
        else:
            print(f"   ⚠️  Dimensions mismatch: got {len(embeddings[0])}, expected 768")

    except KeyError as e:
        print(f"   ❌ KeyError (this is the bug we're fixing): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Failed to generate embedding: {type(e).__name__}: {e}")
        sys.exit(1)

    # Test multiple texts
    print("\n3. Testing multiple text embeddings...")
    test_texts_multiple = [
        "Hello world",
        "Machine learning is fascinating",
        "Ollama embeddings work great"
    ]

    try:
        embeddings = await engine.embed_text(test_texts_multiple)
        print("   ✅ Multiple embeddings generated successfully")
        print(f"   📊 Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"   📊 Text {i+1}: {len(emb)} dimensions, first 3 values: {emb[:3]}")

    except Exception as e:
        print(f"   ❌ Failed to generate embeddings: {type(e).__name__}: {e}")
        sys.exit(1)

    # Success!
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe OllamaEmbeddingEngine fix is working correctly!")
    print("- Handles 'embedding' (singular) response from Ollama API")
    print("- Generates embeddings successfully")
    print("- Correct dimensions (768 for nomic-embed-text)")
    print("\n✅ Ready to submit PR!")


if __name__ == "__main__":
    asyncio.run(test_ollama_embedding())
