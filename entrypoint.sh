#!/bin/bash
# export ENVIRONMENT

echo $DEBUG
echo $ENVIRONMENT

if [ "$ENVIRONMENT" != "local" ]; then
    echo "Running fetch_secret.py"

    python cognitive_architecture/fetch_secret.py

    if [ $? -ne 0 ]; then
        echo "Error: fetch_secret.py failed"
        exit 1
    fi
else
    echo '"local" environment is active, skipping fetch_secret.py'
fi

echo "Running create_database.py"

python cognitive_architecture/database/create_database.py
if [ $? -ne 0 ]; then
    echo "Error: create_database.py failed"
    exit 1
fi

echo "Starting Gunicorn"

if [ "$DEBUG" = true ]; then
  echo "Waiting for the debugger to attach..."
  python -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --bind=0.0.0.0:443 --log-level debug api:app
else
  gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --bind=0.0.0.0:443 --log-level debug api:app
fi
