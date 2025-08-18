#!/usr/bin/env python3
"""
Minimal CLI entry point for cognee that avoids early initialization
"""

import sys
import os
from typing import Any, Sequence

# CRITICAL: Prevent verbose logging initialization for CLI-only usage
# This must be set before any cognee imports to be effective
os.environ["COGNEE_MINIMAL_LOGGING"] = "true"
os.environ["COGNEE_CLI_MODE"] = "true"


def get_version() -> str:
    """Get cognee version without importing the main package"""
    try:
        # Try to get version from pyproject.toml first (for development)
        from pathlib import Path

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("version"):
                        version = line.split("=")[1].strip("'\"\n ")
                        return f"{version}-local"

        # Fallback to installed package version
        import importlib.metadata

        return importlib.metadata.version("cognee")
    except Exception:
        return "unknown"


def get_command_info() -> dict:
    """Get command information without importing cognee"""
    return {
        "add": "Add data to Cognee for knowledge graph processing",
        "search": "Search and query the knowledge graph for insights, information, and connections",
        "cognify": "Transform ingested data into a structured knowledge graph",
        "delete": "Delete data from cognee knowledge base",
        "config": "Manage cognee configuration settings",
    }


def print_help() -> None:
    """Print help message with dynamic command descriptions"""
    commands = get_command_info()
    command_list = "\n".join(f"    {cmd:<12} {desc}" for cmd, desc in commands.items())

    print(f"""
usage: cognee [-h] [--version] [--debug] {{{"|".join(commands.keys())}}} ...

Cognee CLI - Manage your knowledge graphs and cognitive processing pipelines.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --debug               Enable debug mode to show full stack traces on exceptions

Available commands:
  {{{",".join(commands.keys())}}}
{command_list}

For more information on each command, use: cognee <command> --help
""")


def main() -> int:
    """Minimal CLI main function"""
    # Handle help and version without any imports - purely static
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help"]):
        print_help()
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(f"cognee {get_version()}")
        return 0

    # For actual commands, import the full CLI with minimal logging
    try:
        from cognee.cli._cognee import main as full_main

        return full_main()
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print(f"Error: {e}")
        print("Use --debug for full stack trace")
        return 1


if __name__ == "__main__":
    sys.exit(main())
