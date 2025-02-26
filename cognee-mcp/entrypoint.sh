#!/bin/bash

set -e  # Exit on error
echo "Debug mode: $DEBUG"

# Add startup delay to ensure DB is ready
sleep 2

# Modified Gunicorn startup with error handling
if [ $DEBUG == "true" ]; then
    echo "Waiting for the debugger to attach..."
    exec uv run python -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m uv run cognee
else
    exec uv run cognee
fi
