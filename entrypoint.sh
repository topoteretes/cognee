#!/bin/bash

echo "Debug mode: $DEBUG"
echo "Environment: $ENVIRONMENT"

echo "Starting Gunicorn"

if [ "$ENVIRONMENT" = "dev" ]; then
  if [ "$DEBUG" = true ]; then
    echo "Waiting for the debugger to attach..."

    python -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level debug --reload cognee.api.client:app
  else
    gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level debug --reload cognee.api.client:app
  fi
else
  # gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --log-level error cognee.api.client:app
  python ./cognee/api/client.py
fi
