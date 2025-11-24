#!/bin/bash

set -e  # Exit on error
echo "Debug mode: $DEBUG"
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

# Set default transport mode if not specified
TRANSPORT_MODE=${TRANSPORT_MODE:-"stdio"}
echo "Transport mode: $TRANSPORT_MODE"

# Set default ports if not specified
DEBUG_PORT=${DEBUG_PORT:-5678}
HTTP_PORT=${HTTP_PORT:-8000}
echo "Debug port: $DEBUG_PORT"
echo "HTTP port: $HTTP_PORT"

# Check if API mode is enabled
if [ -n "$API_URL" ]; then
    echo "API mode enabled: $API_URL"
    echo "Skipping database migrations (API server handles its own database)"
else
    echo "Direct mode: Using local cognee instance"
    # Run Alembic migrations with proper error handling.
    # Note on UserAlreadyExists error handling:
    # During database migrations, we attempt to create a default user. If this user
    # already exists (e.g., from a previous deployment or migration), it's not a
    # critical error and shouldn't prevent the application from starting. This is
    # different from other migration errors which could indicate database schema
    # inconsistencies and should cause the startup to fail. This check allows for
    # smooth redeployments and container restarts while maintaining data integrity.
    echo "Running database migrations..."

    MIGRATION_OUTPUT=$(alembic upgrade head)
    MIGRATION_EXIT_CODE=$?

    if [[ $MIGRATION_EXIT_CODE -ne 0 ]]; then
        if [[ "$MIGRATION_OUTPUT" == *"UserAlreadyExists"* ]] || [[ "$MIGRATION_OUTPUT" == *"User default_user@example.com already exists"* ]]; then
            echo "Warning: Default user already exists, continuing startup..."
        else
            echo "Migration failed with unexpected error."
            exit 1
        fi
    fi

    echo "Database migrations done."
fi

echo "Starting Cognee MCP Server with transport mode: $TRANSPORT_MODE"

# Add startup delay to ensure DB is ready
sleep 2

# Build API arguments if API_URL is set
API_ARGS=""
if [ -n "$API_URL" ]; then
    # Handle localhost in API_URL - convert to host-accessible address
    if echo "$API_URL" | grep -q "localhost" || echo "$API_URL" | grep -q "127.0.0.1"; then
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
    
    API_ARGS="--api-url $API_URL"
    if [ -n "$API_TOKEN" ]; then
        API_ARGS="$API_ARGS --api-token $API_TOKEN"
    fi
fi

# Modified startup with transport mode selection and error handling
if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "Waiting for the debugger to attach..."
        if [ "$TRANSPORT_MODE" = "sse" ]; then
            exec python -m debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m cognee-mcp --transport sse --host 0.0.0.0 --port $HTTP_PORT --no-migration $API_ARGS
        elif [ "$TRANSPORT_MODE" = "http" ]; then
            exec python -m debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m cognee-mcp --transport http --host 0.0.0.0 --port $HTTP_PORT --no-migration $API_ARGS
        else
            exec python -m debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m cognee-mcp --transport stdio --no-migration $API_ARGS
        fi
    else
        if [ "$TRANSPORT_MODE" = "sse" ]; then
            exec cognee-mcp --transport sse --host 0.0.0.0 --port $HTTP_PORT --no-migration $API_ARGS
        elif [ "$TRANSPORT_MODE" = "http" ]; then
            exec cognee-mcp --transport http --host 0.0.0.0 --port $HTTP_PORT --no-migration $API_ARGS
        else
            exec cognee-mcp --transport stdio --no-migration $API_ARGS
        fi
    fi
else
    if [ "$TRANSPORT_MODE" = "sse" ]; then
        exec cognee-mcp --transport sse --host 0.0.0.0 --port $HTTP_PORT --no-migration $API_ARGS
    elif [ "$TRANSPORT_MODE" = "http" ]; then
        exec cognee-mcp --transport http --host 0.0.0.0 --port $HTTP_PORT --no-migration $API_ARGS
    else
        exec cognee-mcp --transport stdio --no-migration $API_ARGS
    fi
fi
