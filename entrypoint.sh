#!/bin/bash

set -e  # Exit on error
echo "Debug mode: $DEBUG"
echo "Environment: $ENVIRONMENT"

# Run Alembic migrations with proper error handling

# Note on UserAlreadyExists error handling:
# During database migrations, we attempt to create a default user. If this user
# already exists (e.g., from a previous deployment or migration), it's not a
# critical error and shouldn't prevent the application from starting. This is
# different from other migration errors which could indicate database schema
# inconsistencies and should cause the startup to fail. This check allows for
# smooth redeployments and container restarts while maintaining data integrity.
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
    if [ "$DEBUG" = "true" ]; then
        echo "Waiting for the debugger to attach..."
        exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level debug --reload cognee.api.client:app
    else
        exec gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level debug --reload cognee.api.client:app
    fi
else
    exec gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level error cognee.api.client:app 
fi
