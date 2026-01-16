#!/bin/bash

set -e  # Exit on error
echo "Debug mode: $DEBUG"
echo "Environment: $ENVIRONMENT"

# Set default ports if not specified
DEBUG_PORT=${DEBUG_PORT:-5678}
HTTP_PORT=${HTTP_PORT:-8000}
echo "Debug port: $DEBUG_PORT"
echo "HTTP port: $HTTP_PORT"

# Run Alembic migrations with proper error handling.
echo "Running database migrations..."

# Move to the cognee directory to run alembic migrations from there
set +e # Disable exit on error to handle specific migration errors
MIGRATION_OUTPUT=$(cd cognee && alembic upgrade head)
MIGRATION_EXIT_CODE=$?
set -e

if [[ $MIGRATION_EXIT_CODE -ne 0 ]]; then
    if [[ "$MIGRATION_OUTPUT" == *"UserAlreadyExists"* ]] || [[ "$MIGRATION_OUTPUT" == *"User default_user@example.com already exists"* ]]; then
        echo "Warning: Default user already exists, continuing startup..."
    else
        echo "Migration failed with unexpected error. Trying to run Cognee without migrations."

        echo "Initializing database tables..."
        python /app/cognee/modules/engine/operations/setup.py
        INIT_EXIT_CODE=$?

        if [[ $INIT_EXIT_CODE -ne 0 ]]; then
            echo "Database initialization failed!"
            exit 1
        fi
    fi
else
    echo "Database migrations done."
fi

echo "Starting server..."

# Add startup delay to ensure DB is ready
sleep 2

# Modified Gunicorn startup with error handling
if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "Waiting for the debugger to attach..."
        exec debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:$HTTP_PORT --log-level debug --reload --access-logfile - --error-logfile - cognee.api.client:app
    else
        exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:$HTTP_PORT --log-level debug --reload --access-logfile - --error-logfile - cognee.api.client:app
    fi
else
    exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:$HTTP_PORT --log-level error --access-logfile - --error-logfile - cognee.api.client:app
fi
