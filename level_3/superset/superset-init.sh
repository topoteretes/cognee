#!/bin/bash
set -ex

# create Admin user, you can read these values from env or anywhere else possible
superset fab create-admin --username "$ADMIN_USERNAME" --firstname Superset --lastname Admin --email "$ADMIN_EMAIL" --password "$ADMIN_PASSWORD"

# Upgrading Superset metastore
superset db upgrade

# setup roles and permissions
superset superset init

# Starting server in the background
/bin/sh -c /usr/bin/run-server.sh &

# Waiting for the server to start
sleep 15

## Running the script to add database connections
#python /app/add_database_connections.py

# Bring the server process back into the foreground so that the script doesn't exit and the server keeps running.
wait $!