try:
    from .server import main as server_main
except ImportError:
    from server import main as server_main


def main_mcp():
    """Entry point for cognee-memory command."""
    import asyncio

    asyncio.run(server_main())
