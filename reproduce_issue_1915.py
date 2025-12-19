import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.append(os.getcwd())

# Configure logging to see warnings/errors
logging.basicConfig(level=logging.INFO)

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer

def test_tokenizer_model_name():
    print("--- Starting Reproduction Test ---")
    
    # Scenario: provider="custom", model="openai/BAAI/bge-m3"
    # This simulates what the user reports using.
    engine = LiteLLMEmbeddingEngine(
        model="openai/BAAI/bge-m3",
        provider="custom",
        api_key="fake",
        endpoint="fake"
    )
    
    tokenizer = engine.get_tokenizer()
    
    print(f"Provider: {engine.provider}")
    print(f"Model Input: {engine.model}")
    print(f"Resulting Tokenizer: {type(tokenizer).__name__}")
    
    if hasattr(tokenizer, 'model'):
        print(f"Tokenizer Model: {tokenizer.model}")
        
    # We expect or want HuggingFaceTokenizer with model="BAAI/bge-m3"
    
    if isinstance(tokenizer, HuggingFaceTokenizer):
        if tokenizer.model == "BAAI/bge-m3":
            print("SUCCESS: Tokenizer model is 'BAAI/bge-m3'")
        else:
            print(f"FAILURE: Tokenizer model is '{tokenizer.model}' (Expected 'BAAI/bge-m3')")
    
    elif isinstance(tokenizer, TikTokenTokenizer):
         # If it fell back to TikToken, that means HF failed (likely because 'openai/BAAI/bge-m3' was passed)
         print("FAILURE: Fell back to TikToken (likely due to HF loading error with 'openai/BAAI/bge-m3')")

if __name__ == "__main__":
    test_tokenizer_model_name()
