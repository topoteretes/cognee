from .server import main as server_main


def main():
    """Main entry point for the package."""
    import asyncio

    asyncio.run(server_main())
