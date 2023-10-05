#!/bin/bash
export ENVIRONMENT
python fetch_secret.py

# Start Gunicorn
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -t 120 --bind=0.0.0.0:8000 --bind=0.0.0.0:443 --log-level debug  api:app