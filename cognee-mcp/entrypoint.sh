#!/bin/bash

set -e  # Exit on error
echo "Debug mode: $DEBUG"
echo "Environment: $ENVIRONMENT"

# Set default transport mode if not specified
TRANSPORT_MODE=${TRANSPORT_MODE:-"stdio"}
echo "Transport mode: $TRANSPORT_MODE"

# Set default ports if not specified
DEBUG_PORT=${DEBUG_PORT:-5678}
HTTP_PORT=${HTTP_PORT:-8000}
echo "Debug port: $DEBUG_PORT"
echo "HTTP port: $HTTP_PORT"

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

echo "Starting Cognee MCP Server with transport mode: $TRANSPORT_MODE"

# Add startup delay to ensure DB is ready
sleep 2

# Modified startup with transport mode selection and error handling
if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "Waiting for the debugger to attach..."
        if [ "$TRANSPORT_MODE" = "sse" ]; then
            exec python -m debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m cognee-mcp --transport sse --host 0.0.0.0 --port $HTTP_PORT --no-migration
        elif [ "$TRANSPORT_MODE" = "http" ]; then
            exec python -m debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m cognee-mcp --transport http --host 0.0.0.0 --port $HTTP_PORT --no-migration
        else
            exec python -m debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m cognee-mcp --transport stdio --no-migration
        fi
    else
        if [ "$TRANSPORT_MODE" = "sse" ]; then
            exec cognee-mcp --transport sse --host 0.0.0.0 --port $HTTP_PORT --no-migration
        elif [ "$TRANSPORT_MODE" = "http" ]; then
            exec cognee-mcp --transport http --host 0.0.0.0 --port $HTTP_PORT --no-migration
        else
            exec cognee-mcp --transport stdio --no-migration
        fi
    fi
else
    if [ "$TRANSPORT_MODE" = "sse" ]; then
        exec cognee-mcp --transport sse --host 0.0.0.0 --port $HTTP_PORT --no-migration
    elif [ "$TRANSPORT_MODE" = "http" ]; then
        exec cognee-mcp --transport http --host 0.0.0.0 --port $HTTP_PORT --no-migration
    else
        exec cognee-mcp --transport stdio --no-migration
    fi
fi
