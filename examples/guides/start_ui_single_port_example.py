#!/usr/bin/env python3
"""
Example showing how to serve cognee's UI, API backend and MCP server behind a single port.

By default start_ui() gives each of the three services its own port, which means a
deployment with only one domain available cannot expose all of them. Passing gateway_port
puts a gateway in front: it is the only publicly bound listener, and it routes to the
services by path prefix. No external reverse proxy is involved.
"""

import asyncio
import time

import cognee

GATEWAY_PORT = 9000


async def main():
    print("Adding sample data to cognee...")
    await cognee.remember(
        "Cognee turns raw data into a knowledge graph that AI agents can query.",
        self_improvement=False,
    )

    print("\n" + "=" * 60)
    print(f"Starting cognee behind a single port ({GATEWAY_PORT})...")
    print("=" * 60)

    def dummy_callback(pid):
        pass

    server = cognee.start_ui(
        pid_callback=dummy_callback,
        start_backend=True,
        start_mcp=True,
        gateway_port=GATEWAY_PORT,
        open_browser=True,
    )

    if not server:
        print("Failed to start the UI. Check the logs above for details.")
        return

    print("Everything is reachable on one port:")
    print(f"  UI       http://localhost:{GATEWAY_PORT}/")
    print(f"  API      http://localhost:{GATEWAY_PORT}/backend")
    print(f"  MCP      http://localhost:{GATEWAY_PORT}/mcp")
    print("\nOnly this port needs to be publicly reachable — the frontend, backend and")
    print("MCP server stay bound to localhost behind it.")
    print("\nPress Ctrl+C to stop...")

    try:
        while server.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        server.terminate()
        server.wait()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
