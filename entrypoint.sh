#!/bin/bash

set -e  # Exit on error

# Ensure the Cognee service uses its dedicated LiteLLM key when one is provided
if [ -n "$COGNEE_API_KEY" ]; then
  export LLM_API_KEY="$COGNEE_API_KEY"
  export OPENAI_API_KEY="$COGNEE_API_KEY"
  export LITELLM_PROXY_API_KEY="${LITELLM_PROXY_API_KEY:-$COGNEE_API_KEY}"
fi

if [ -z "$DATA_ROOT_DIRECTORY" ] && [ "${STORAGE_BACKEND:-s3}" = "s3" ]; then
  bucket="${COGNEE_S3_BUCKET:-projects}"
  prefix="${COGNEE_S3_PREFIX:-}"
  prefix="${prefix#/}"
  prefix="${prefix%/}"
  root="s3://$bucket"
  if [ -n "$prefix" ]; then
    root="$root/$prefix"
  fi
  export DATA_ROOT_DIRECTORY="$root"
fi

if [ -z "$SYSTEM_ROOT_DIRECTORY" ]; then
  export SYSTEM_ROOT_DIRECTORY="${DATA_ROOT_DIRECTORY:-/data/storage}"
fi

if [ -z "$CACHE_ROOT_DIRECTORY" ]; then
  export CACHE_ROOT_DIRECTORY="${SYSTEM_ROOT_DIRECTORY}/cache"
fi
echo "Debug mode: $DEBUG"
echo "Environment: $ENVIRONMENT"

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

echo "Starting server..."

# Add startup delay to ensure DB is ready
sleep 2

# Modified Gunicorn startup with error handling
if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "local" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "Waiting for the debugger to attach..."
        debugpy --wait-for-client --listen 0.0.0.0:$DEBUG_PORT -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:$HTTP_PORT --log-level debug --reload cognee.api.client:app
    else
        gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:$HTTP_PORT --log-level debug --reload cognee.api.client:app
    fi
else
    gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:$HTTP_PORT --log-level error cognee.api.client:app
fi
