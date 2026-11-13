import importlib.metadata
import os
from contextlib import suppress
from pathlib import Path


def get_cognee_version() -> str:
    """Returns either the version of installed cognee package or the one
    found in nearby pyproject.toml.

    A copy of the package that has neither (vendored source, a bundled app, a
    layer that strips *.dist-info) falls back to the version the build backend
    stamped into ``cognee/_version.py``. Only when that file is missing too is
    the version ``unknown``."""
    with (
        suppress(FileNotFoundError, StopIteration),
        open(
            os.path.join(Path(__file__).parent.parent, "pyproject.toml"), encoding="utf-8"
        ) as pyproject_toml,
    ):
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
        pass
    try:
        from cognee._version import __version__
    except ImportError:
        return "unknown"
    return __version__
