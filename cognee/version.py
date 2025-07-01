import os
from contextlib import suppress
import importlib.metadata
from pathlib import Path


def get_cognee_version() -> str:
    """Returns either the version of installed cognee package or the one
    found in nearby pyproject.toml"""
    with suppress(FileNotFoundError, StopIteration):
        with open(
            os.path.join(Path(__file__).parent.parent, "pyproject.toml"), encoding="utf-8"
        ) as pyproject_toml:
            version = (
                next(line for line in pyproject_toml if line.startswith("version"))
                .split("=")[1]
                .strip("'\"\n ")
            )
            # Mark the version as a local Cognee library by appending “-dev”
            return f"{version}-local"
    try:
        return importlib.metadata.version("cognee")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
