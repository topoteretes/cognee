#!/usr/bin/env bash

echo "$@"

exec "$@" # Runs the command passed to the entrypoint script.
