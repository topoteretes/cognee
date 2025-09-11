#!/usr/bin/env python3
"""
Example showing how to use cognee.start_ui() to launch the frontend.

This demonstrates the new UI functionality that works similar to DuckDB's start_ui().
"""

import asyncio
import cognee
import time


async def main():
    # First, let's add some data to cognee for the UI to display
    print("Adding sample data to cognee...")
    await cognee.add(
        "Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval."
    )
    await cognee.add(
        "Machine learning (ML) is a subset of artificial intelligence that focuses on algorithms and statistical models."
    )

    # Generate the knowledge graph
    print("Generating knowledge graph...")
    await cognee.cognify()

    print("\n" + "=" * 60)
    print("Starting cognee UI...")
    print("=" * 60)

    # Start the UI server
    server = cognee.start_ui(
        host="localhost",
        port=3000,
        open_browser=True,  # This will automatically open your browser
    )

    if server:
        print("UI server started successfully!")
        print("The interface will be available at: http://localhost:3000")
        print("\nPress Ctrl+C to stop the server when you're done...")

        try:
            # Keep the server running
            while server.poll() is None:  # While process is still running
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping UI server...")
            server.terminate()
            server.wait()  # Wait for process to finish
            print("UI server stopped.")
    else:
        print("Failed to start UI server. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
