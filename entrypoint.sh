#!/bin/bash
export ENVIRONMENT
# Run Python scripts with error handling
echo "Running fetch_secret.py"
python cognitive_architecture/fetch_secret.py
if [ $? -ne 0 ]; then
    echo "Error: fetch_secret.py failed"
    exit 1
fi

echo "Running create_database.py"
python cognitive_architecture/database/create_database.py
if [ $? -ne 0 ]; then
    echo "Error: create_database.py failed"
    exit 1
fi

# Start Gunicorn
echo "Starting Gunicorn"
gunicorn -w 3 -k uvicorn.workers.UvicornWorker -t 30000 --bind=0.0.0.0:8000 --bind=0.0.0.0:443 --log-level debug api:app
