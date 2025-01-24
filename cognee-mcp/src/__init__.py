from .server import mcp


def main():
    """Main entry point for the package."""
    mcp.run(transport="stdio")
