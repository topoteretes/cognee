#!/bin/bash

set -e  # Exit on error
# ENV is the canonical environment variable (matches what the app reads).
# ENVIRONMENT is accepted as a deprecated fallback for older deployments.
ENV="${ENV:-${ENVIRONMENT:-}}"
echo "Debug mode: $DEBUG"
echo "Environment: $ENV"

# Set default ports and bind address if not specified
DEBUG_PORT=${DEBUG_PORT:-5678}
HTTP_PORT=${HTTP_PORT:-8000}
BIND_ADDRESS=${BIND_ADDRESS:-"0.0.0.0"}
echo "Debug port: $DEBUG_PORT"
echo "HTTP port: $HTTP_PORT"
echo "Bind address: $BIND_ADDRESS"

# Run migrations through cognee's own migration system rather than raw
# alembic: it knows a fresh database from an existing one (fresh -> create
# directories + build the schema from the models + `alembic stamp head`;
# existing -> apply Alembic deltas + the graph/vector data chain), honors
# ENABLE_AUTO_MIGRATIONS, and needs no working directory. Raw
# `alembic upgrade head` had none of that: on a fresh volume it died on the
# missing databases directory and the silent create_all fallback left the
# schema unstamped.
#
# A relational migration failure aborts the boot (set -e). Per-dataset
# data-chain failures do not: the server comes up, cognee blocks writes to
# just those datasets and retries on the next start — the same semantics as
# the in-app startup path.
echo "Running database migrations..."
python <<'PYTHON'
import asyncio

from cognee.modules.migrations.startup import run_migrations

failed = asyncio.run(run_migrations())
if failed:
    print(
        "Data migrations failed for: " + ", ".join(failed)
        + ". Writes to those datasets are blocked until they migrate; "
        "retried on the next start."
    )
PYTHON
echo "Database migrations done."

echo "Starting server..."

# Add startup delay to ensure DB is ready
sleep 2

# Modified Gunicorn startup with error handling
if [ "$ENV" = "dev" ] || [ "$ENV" = "local" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "Waiting for the debugger to attach..."
        exec debugpy --wait-for-client --listen $BIND_ADDRESS:$DEBUG_PORT -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=$BIND_ADDRESS:$HTTP_PORT --log-level debug --reload --access-logfile - --error-logfile - cognee.api.client:app
    else
        exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=$BIND_ADDRESS:$HTTP_PORT --log-level debug --reload --access-logfile - --error-logfile - cognee.api.client:app
    fi
else
    exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -t 30000 --bind=$BIND_ADDRESS:$HTTP_PORT --log-level error --access-logfile - --error-logfile - cognee.api.client:app
fi
