#!/bin/bash

set -e  # Exit on error
echo "Environment: $ENVIRONMENT"

# Install optional dependencies if EXTRAS is set
if [ -n "$EXTRAS" ]; then
    echo "Installing optional dependencies: $EXTRAS"

    # Get the cognee version that's currently installed
    COGNEE_VERSION=$(uv pip show cognee | grep "Version:" | awk '{print $2}')
    echo "Current cognee version: $COGNEE_VERSION"

    # Build the extras list for cognee
    IFS=',' read -ra EXTRA_ARRAY <<< "$EXTRAS"
    # Combine base extras from pyproject.toml with requested extras
    ALL_EXTRAS=""
    for extra in "${EXTRA_ARRAY[@]}"; do
        # Trim whitespace
        extra=$(echo "$extra" | xargs)
        # Add to extras list if not already present
        if [[ ! "$ALL_EXTRAS" =~ (^|,)"$extra"(,|$) ]]; then
            if [ -z "$ALL_EXTRAS" ]; then
                ALL_EXTRAS="$extra"
            else
                ALL_EXTRAS="$ALL_EXTRAS,$extra"
            fi
        fi
    done

    echo "Installing cognee with extras: $ALL_EXTRAS"
    echo "Running: uv pip install 'cognee[$ALL_EXTRAS]==$COGNEE_VERSION'"
    uv pip install "cognee[$ALL_EXTRAS]==$COGNEE_VERSION"

    # Verify installation
    echo ""
    echo "✓ Optional dependencies installation completed"
else
    echo "No optional dependencies specified"
fi

ARGS="$@" #forward any args passed to the container at runtime
echo "$ARGS"

# Set default transport mode if not specified
TRANSPORT_MODE=${TRANSPORT_MODE:-"stdio"}
echo "Transport mode: $TRANSPORT_MODE"

# Set default ports if not specified

if [ "$TRANSPORT_MODE" != "stdio" ]; then
    HTTP_PORT=${HTTP_PORT:-8000}
    echo "HTTP port: $HTTP_PORT"
    ARGS="$ARGS --host 0.0.0.0 --port $HTTP_PORT"
fi

echo "Starting Cognee MCP Server with transport mode: $TRANSPORT_MODE"

# Add startup delay to ensure DB is ready
sleep 2

# Build API arguments if API_URL is set
if [ -n "$API_URL" ]; then
    echo "API mode enabled: $API_URL"

    # Handle localhost in API_URL - convert to host-accessible address
    if [[ "$API_URL" =~ "localhost" || "$API_URL" =~ "127.0.0.1" ]]; then
        echo "⚠️  Warning: API_URL contains localhost/127.0.0.1"
        echo "   Original: $API_URL"

        # Try to use host.docker.internal (works on Mac/Windows and recent Linux with Docker Desktop)
        FIXED_API_URL=$(echo "$API_URL" | sed 's/localhost/host.docker.internal/g' | sed 's/127\.0\.0\.1/host.docker.internal/g')

        echo "   Converted to: $FIXED_API_URL"
        echo "   This will work on Mac/Windows/Docker Desktop."
        echo "   On Linux without Docker Desktop, you may need to:"
        echo "     - Use --network host, OR"
        echo "     - Set API_URL=http://172.17.0.1:8000 (Docker bridge IP)"

        API_URL="$FIXED_API_URL"
    fi

    ARGS="$ARGS --api-url $API_URL"
    if [ -n "$API_TOKEN" ]; then
        ARGS="$ARGS --api-token $API_TOKEN"
    fi
else
    echo "Direct mode: Using local cognee instance"
fi

#ARGS needs to not be quoted here. It contains multiple values that should be treated as separate arguments
if [ "$DEBUG" = "true" ] && [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; then    
    DEBUG_PORT=${DEBUG_PORT:-5678}
    echo "Running in debug mode"
    echo "Debug port: $DEBUG_PORT"
    echo "Waiting for the debugger to attach..."
    exec python -m debugpy --wait-for-client --listen 0.0.0.0:"$DEBUG_PORT" -m cognee-mcp --transport "$TRANSPORT_MODE" $ARGS
else
    exec cognee-mcp --transport "$TRANSPORT_MODE" $ARGS
fi