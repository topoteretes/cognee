import sys
import argparse
import asyncio
import subprocess
from pathlib import Path

from cognee.shared.logging_utils import get_logger, setup_logging
from mcp.server import FastMCP
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

from src.shared import context
from src.clients import CogneeClient
from src.tools import (
    cognify,
    search,
    list_data,
    delete,
    prune,
    cognify_status,
)


mcp = FastMCP("Cognee")

logger = get_logger()

mcp.tool()(cognify)
mcp.tool()(search)
mcp.tool()(list_data)
mcp.tool()(delete)
mcp.tool()(prune)
mcp.tool()(cognify_status)


async def run_sse_with_cors():
    """Custom SSE transport with CORS middleware."""
    sse_app = mcp.sse_app()
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        sse_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_http_with_cors():
    """Custom HTTP transport with CORS middleware."""
    http_app = mcp.streamable_http_app()
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        http_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok"})


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    # HTTP transport options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server to (default: 8000)",
    )

    parser.add_argument(
        "--path",
        default="/mcp",
        help="Path for the MCP HTTP endpoint (default: /mcp)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for the HTTP server (default: info)",
    )

    parser.add_argument(
        "--no-migration",
        default=False,
        action="store_true",
        help="Argument stops database migration from being attempted",
    )

    # Cognee API connection options
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL of a running Cognee FastAPI server (e.g., http://localhost:8000). "
        "If provided, the MCP server will connect to the API instead of using cognee directly.",
    )

    parser.add_argument(
        "--api-token",
        default=None,
        help="Authentication token for the API (optional, required if API has authentication enabled).",
    )

    args = parser.parse_args()

    # Initialize the global CogneeClient
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)
    context.set_cognee_client(cognee_client)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Skip migrations when in API mode (the API server handles its own database)
    if not args.no_migration and not args.api_url:
        # Run Alembic migrations from the main cognee directory where alembic.ini is located
        logger.info("Running database migrations...")
        migration_result = subprocess.run(
            ["python", "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )

        if migration_result.returncode != 0:
            migration_output = migration_result.stderr + migration_result.stdout
            # Check for the expected UserAlreadyExists error (which is not critical)
            if (
                "UserAlreadyExists" in migration_output
                or "User default_user@example.com already exists" in migration_output
            ):
                logger.warning("Warning: Default user already exists, continuing startup...")
            else:
                logger.error(f"Migration failed with unexpected error: {migration_output}")
                sys.exit(1)

        logger.info("Database migrations done.")
    elif args.api_url:
        logger.info("Skipping database migrations (using API mode)")

    logger.info(f"Starting MCP server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(f"Running MCP server with SSE transport on {args.host}:{args.port}")
        await run_sse_with_cors()
    elif args.transport == "http":
        logger.info(
            f"Running MCP server with Streamable HTTP transport on {args.host}:{args.port}{args.path}"
        )
        await run_http_with_cors()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
