"""
Module to suppress verbose logging before any cognee imports.
This must be imported before any other cognee modules.
"""

import os

# Set CLI mode to suppress verbose logging
os.environ["COGNEE_CLI_MODE"] = "true"

# Also set log level to ERROR for extra safety
os.environ["LOG_LEVEL"] = "ERROR"
