#!/bin/bash
# Cognee MCP Server - Optimized Entrypoint
# Production-ready entrypoint with proper error handling and validation

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ================================
# Configuration & Defaults
# ================================

DEBUG=${DEBUG:-false}
ENVIRONMENT=${ENVIRONMENT:-production}
TRANSPORT_MODE=${TRANSPORT_MODE:-stdio}
DEBUG_PORT=${DEBUG_PORT:-5678}
HTTP_PORT=${HTTP_PORT:-8000}
API_URL=${API_URL:-}
API_TOKEN=${API_TOKEN:-}
EXTRAS=${EXTRAS:-}

echo "========================================"
echo "Cognee MCP Server Starting"
echo "========================================"
echo "Environment: $ENVIRONMENT"
echo "Debug mode: $DEBUG"
echo "Transport mode: $TRANSPORT_MODE"
echo "HTTP port: $HTTP_PORT"
echo "Debug port: $DEBUG_PORT"
echo "========================================"

# ================================
# Validation
# ================================

# Validate API_URL is set (required for MCP server)
if [ -z "$API_URL" ]; then
    echo "ERROR: API_URL is required"
    echo "The MCP server must connect to a running Cognee API server."
    echo "Set API_URL environment variable (e.g., http://cognee-api:8000)"
    exit 1
fi

echo "✓ API mode enabled: $API_URL"

# Validate transport mode
case "$TRANSPORT_MODE" in
    stdio|sse|http)
        echo "✓ Valid transport mode: $TRANSPORT_MODE"
        ;;
    *)
        echo "ERROR: Invalid TRANSPORT_MODE: $TRANSPORT_MODE"
        echo "Must be one of: stdio, sse, http"
        exit 1
        ;;
esac

# ================================
# Optional Dependencies Installation
# ================================

if [ -n "$EXTRAS" ]; then
    echo "Installing optional dependencies: $EXTRAS"

    # Get current cognee version
    COGNEE_VERSION=$(uv pip show cognee 2>/dev/null | grep "Version:" | awk '{print $2}')

    if [ -z "$COGNEE_VERSION" ]; then
        echo "ERROR: Could not detect cognee version"
        exit 1
    fi

    echo "Current cognee version: $COGNEE_VERSION"

    # Build extras list
    IFS=',' read -ra EXTRA_ARRAY <<< "$EXTRAS"
    ALL_EXTRAS=""
    for extra in "${EXTRA_ARRAY[@]}"; do
        extra=$(echo "$extra" | xargs)  # Trim whitespace
        if [ -n "$extra" ]; then
            if [ -z "$ALL_EXTRAS" ]; then
                ALL_EXTRAS="$extra"
            else
                ALL_EXTRAS="$ALL_EXTRAS,$extra"
            fi
        fi
    done

    if [ -n "$ALL_EXTRAS" ]; then
        echo "Installing cognee with extras: $ALL_EXTRAS"
        if ! uv pip install "cognee[$ALL_EXTRAS]==$COGNEE_VERSION"; then
            echo "ERROR: Failed to install optional dependencies"
            exit 1
        fi
        echo "✓ Optional dependencies installed successfully"
    fi
else
    echo "No optional dependencies specified (EXTRAS not set)"
fi

# ================================
# API URL Processing
# ================================

# Handle localhost in API_URL - convert to Docker-accessible address
if echo "$API_URL" | grep -qE "(localhost|127\.0\.0\.1)"; then
    echo "⚠️  Warning: API_URL contains localhost/127.0.0.1"
    echo "   Original: $API_URL"

    # Convert to host.docker.internal (works on Mac/Windows/Docker Desktop)
    FIXED_API_URL=$(echo "$API_URL" | \
        sed 's/localhost/host.docker.internal/g' | \
        sed 's/127\.0\.0\.1/host.docker.internal/g')

    echo "   Converted to: $FIXED_API_URL"
    echo "   Note: This works on Mac/Windows/Docker Desktop."
    echo "         On Linux without Docker Desktop, use:"
    echo "         - --network host, OR"
    echo "         - API_URL=http://172.17.0.1:8000 (Docker bridge IP)"

    API_URL="$FIXED_API_URL"
fi

echo "✓ API URL configured: $API_URL"

# ================================
# Build Command Arguments
# ================================

# Build API arguments
API_ARGS="--api-url $API_URL"
if [ -n "$API_TOKEN" ]; then
    API_ARGS="$API_ARGS --api-token $API_TOKEN"
    echo "✓ API token configured"
else
    echo "ℹ️  No API token provided (optional)"
fi

# Build transport arguments
TRANSPORT_ARGS="--transport $TRANSPORT_MODE"

if [ "$TRANSPORT_MODE" = "sse" ] || [ "$TRANSPORT_MODE" = "http" ]; then
    TRANSPORT_ARGS="$TRANSPORT_ARGS --host 0.0.0.0 --port $HTTP_PORT"
    echo "✓ Network transport configured: 0.0.0.0:$HTTP_PORT"
fi

# Combine all arguments
CMD_ARGS="$TRANSPORT_ARGS $API_ARGS"

# ================================
# Startup Delay
# ================================

# Give backend time to initialize
echo "Waiting 2 seconds for backend readiness..."
sleep 2

# ================================
# Server Startup
# ================================

echo "========================================"
echo "Starting Cognee MCP Server"
echo "Command: cognee-mcp $CMD_ARGS"
echo "========================================"

# Start server based on environment and debug mode
if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "🐛 Debug mode enabled - waiting for debugger on port $DEBUG_PORT"
        exec python -m debugpy \
            --wait-for-client \
            --listen 0.0.0.0:$DEBUG_PORT \
            -m src.server $CMD_ARGS
    else
        exec cognee-mcp $CMD_ARGS
    fi
else
    # Production mode
    exec cognee-mcp $CMD_ARGS
fi
