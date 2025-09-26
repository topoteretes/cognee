from .server import main as server_main
import warnings


def main():
    """Deprecated main entry point for the package."""
    import asyncio
    
    # Show deprecation warning
    warnings.warn(
        "The 'cognee' command for cognee-mcp is deprecated and will be removed in a future version. "
        "Please use 'cognee-mcp' instead to avoid conflicts with the main cognee library.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("⚠️  DEPRECATION WARNING: Use 'cognee-mcp' command instead of 'cognee'")
    print("   This avoids conflicts with the main cognee library.")
    print()

    asyncio.run(server_main())


def main_mcp():
    """Clean main entry point for cognee-mcp command."""
    import asyncio
    asyncio.run(server_main())
