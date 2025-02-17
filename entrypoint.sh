#!/bin/bash

set -e  # Exit on error
echo "Debug mode: $DEBUG"
echo "Environment: $ENVIRONMENT"

# Run Alembic migrations with proper error handling
echo "Running database migrations..."
MIGRATION_OUTPUT=$(poetry run alembic upgrade head 2>&1) || {
    if [[ $MIGRATION_OUTPUT == *"UserAlreadyExists"* ]] || [[ $MIGRATION_OUTPUT == *"User default_user@example.com already exists"* ]]; then
        echo "Warning: Default user already exists, continuing startup..."
    else
        echo "Migration failed with unexpected error:"
        echo "$MIGRATION_OUTPUT"
        exit 1
    fi
}

echo "Starting Gunicorn"

# Add startup delay to ensure DB is ready
sleep 2

# Modified Gunicorn startup with error handling
if [ "$ENVIRONMENT" = "dev" ]; then
    if [ "$DEBUG" = true ]; then
        echo "Waiting for the debugger to attach..."
        exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level debug --reload cognee.api.client:app
    else
        exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level debug --reload cognee.api.client:app
    fi
else
    exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level error cognee.api.client:app 
fi