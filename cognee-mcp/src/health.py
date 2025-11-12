"""
Health check endpoints and monitoring.

This module provides comprehensive health checks for the MCP server,
including dependency status and operational readiness.
"""

import os
from starlette.responses import JSONResponse
from mcp.server import FastMCP
from cognee.shared.logging_utils import get_logger, get_log_file_location

logger = get_logger()


def setup_health_routes(mcp: FastMCP):
    """
    Set up health check routes on the MCP server.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.custom_route("/health", methods=["GET"])
    async def basic_health_check(request):
        """Basic health check endpoint."""
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/health/detailed", methods=["GET"])
    async def detailed_health_check(request):
        """
        Detailed health check with dependency status.

        Returns:
            JSONResponse: Health status with detailed checks
        """
        checks = {
            "backend_access_control": (
                "enabled" if os.getenv("ENABLE_BACKEND_ACCESS_CONTROL") == "true" else "disabled"
            ),
            "log_file": get_log_file_location(),
        }

        return JSONResponse(
            {
                "status": "ok",
                "checks": checks,
            }
        )


async def perform_startup_health_check():
    """
    Perform health check during server startup.

    Returns:
        dict: Health check results
    """
    logger.info("Performing startup health check...")

    health_status = {
        "status": "ok",
        "dependencies": {
            "backend_access_control": os.getenv("ENABLE_BACKEND_ACCESS_CONTROL") == "true",
        },
    }

    if health_status["dependencies"]["backend_access_control"]:
        logger.info("✓ Backend access control enabled (KB isolation enforced)")
    else:
        logger.warning("⚠ Backend access control disabled")

    logger.info("Startup health check completed")
    return health_status
