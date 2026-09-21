import importlib.metadata
from contextlib import suppress
from pathlib import Path

_PYPROJECT_TOML = Path(__file__).parent.parent / "pyproject.toml"


def is_source_checkout() -> bool:
    """True when cognee is imported from a tree that carries its ``pyproject.toml``.

    That is a git checkout, but also any image that copies the project file next
    to the package (the official Dockerfile does), which is why the ``-local``
    version suffix built on this rule cannot tell the two apart. Telemetry's
    ``install_kind`` (``cognee.shared.utils.get_install_kind``) layers explicit
    signals on top of it.
    """
    return _PYPROJECT_TOML.is_file()


def get_cognee_version() -> str:
    """Returns either the version of installed cognee package or the one
    found in nearby pyproject.toml"""
    with (
        suppress(FileNotFoundError, StopIteration),
        open(_PYPROJECT_TOML, encoding="utf-8") as pyproject_toml,
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
        return "unknown"
