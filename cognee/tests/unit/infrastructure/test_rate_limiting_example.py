import asyncio
import time
import cognee
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.search import SearchType
from cognee.infrastructure.llm.rate_limiter import LLMRateLimiter
from cognee.infrastructure.llm.config import get_llm_config


async def test_rate_limiting():
    """Test the rate limiting feature with 60 requests per minute."""
    print("\n=== Testing Rate Limiting Feature ===")

    # Configure rate limiting
    print("Configuring rate limiting: 60 requests per minute")
    import os

    os.environ["LLM_RATE_LIMIT_ENABLED"] = "true"
    os.environ["LLM_RATE_LIMIT_REQUESTS"] = "60"
    os.environ["LLM_RATE_LIMIT_INTERVAL"] = "60"

    # Create a fresh limiter instance
    limiter = LLMRateLimiter()
    config = get_llm_config()
    print(
        f"Rate limit settings: {config.llm_rate_limit_enabled=}, {config.llm_rate_limit_requests=}, {config.llm_rate_limit_interval=}"
    )

    # Track successful and failed requests
    successes = []
    failures = []

    print("Making 70 test requests (expecting ~60 to succeed)...")
    start_time = time.time()

    # Try 70 requests (more than our limit of 60)
    for i in range(70):
        if limiter.hit_limit():
            successes.append(i)
        else:
            failures.append(i)

    end_time = time.time()
    elapsed = end_time - start_time

    # Print results
    print(f"Test completed in {elapsed:.2f} seconds")
    print(f"Successful requests: {len(successes)}")
    print(f"Failed requests: {len(failures)}")

    if failures:
        print(f"First failure occurred at request #{failures[0]}")

    # Calculate effective rate
    rate_per_minute = len(successes) / (elapsed / 60)
    print(f"Effective rate: {rate_per_minute:.1f} requests per minute")

    # Verify results
    if 58 <= len(successes) <= 62:
        print("✅ PASS: Rate limiting correctly allowed ~60 requests")
    else:
        print(f"❌ FAIL: Expected ~60 successful requests, got {len(successes)}")

    if len(failures) > 0:
        print("✅ PASS: Rate limiting correctly blocked excess requests")
    else:
        print("❌ FAIL: Expected some requests to be rate-limited")

    print("=== Rate Limiting Test Complete ===\n")


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    print("Adding text to cognee:")
    print(text.strip())
    # Add the text, and make it available for cognify
    await cognee.add(text)
    print("Text added successfully.\n")

    print("Running cognify to create knowledge graph...\n")
    print("Cognify process steps:")
    print("1. Classifying the document: Determining the type and category of the input text.")
    print(
        "2. Checking permissions: Ensuring the user has the necessary rights to process the text."
    )
    print(
        "3. Extracting text chunks: Breaking down the text into sentences or phrases for analysis."
    )
    print("4. Adding data points: Storing the extracted chunks for processing.")
    print(
        "5. Generating knowledge graph: Extracting entities and relationships to form a knowledge graph."
    )
    print("6. Summarizing text: Creating concise summaries of the content for quick insights.\n")

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()
    print("Cognify process complete.\n")

    query_text = "Tell me about NLP"
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(query_type=SearchType.INSIGHTS, query_text=query_text)

    print("Search results:")
    # Display results
    for result_text in search_results:
        print(result_text)

    # Run the rate limiting test after the regular example
    await test_rate_limiting()


if __name__ == "__main__":
    logger = get_logger()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
