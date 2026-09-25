"""Find and load the one ``.env`` file cognee runs with.

Cognee reads settings two ways: settings classes (``BaseSettings`` with
``env_file=".env"``) and plain ``os.getenv`` at the point of use. Both must see
the same file with the same precedence, so the file is resolved here once and
loaded into the process environment with ``override=True`` — the file wins over
variables already present in the shell, which is what users expect from a
project ``.env`` and what preset shell variables kept breaking.

Where the file comes from, in order:

1. ``COGNEE_ENV_FILE``, when set: that exact file, no search. A path that is not
   a file is an error rather than a silent fall back to the search.
2. ``.env`` in the working directory or any parent up to the project root — the
   project.
3. ``.env`` in the cognee package directory or any parent up to the project
   root — where a bare ``load_dotenv()`` used to look. Kept so an environment
   that lives inside the project (``python -m venv .venv`` at the project root)
   keeps working when a script is run from elsewhere.
4. Nothing: the process environment is used as is.

The project root is the nearest directory holding a ``.git`` or
``pyproject.toml``; a ``.env`` above it belongs to someone else and is not
loaded. Without any marker (a plain folder of scripts), the search continues up
to the filesystem root, so a simple script's ``.env`` is still found from a
subdirectory.

The search is explicit, so it does not depend on python-dotenv's guesses
about REPLs, notebooks or debuggers, which made the old behaviour differ
between a script, a notebook and the same script under a debugger. The load
happens once per process, at import; the resolved path is logged once logging
is configured (see ``cognee/__init__.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import dotenv

ENV_FILE_VARIABLE = "COGNEE_ENV_FILE"
PROJECT_MARKERS = (".git", "pyproject.toml")

_loaded = False
_resolved: str | None = None


def _project_root(start: Path) -> Path | None:
    """The nearest of ``start`` and its parents holding a project marker, or None."""
    for directory in (start, *start.parents):
        if any((directory / marker).exists() for marker in PROJECT_MARKERS):
            return directory
    return None


def _find_upward(start: Path, filename: str = ".env") -> str | None:
    """The first ``filename`` in ``start`` or a parent up to the project root (if any), or None."""
    try:
        start = start.resolve()
    except OSError:
        return None
    root = _project_root(start)
    for directory in (start, *start.parents):
        candidate = directory / filename
        if candidate.is_file():
            return str(candidate)
        if directory == root:
            break
    return None


def _explicit_env_file() -> str | None:
    """The file named by ``COGNEE_ENV_FILE``, or None when it is unset or empty."""
    value = os.environ.get(ENV_FILE_VARIABLE, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{ENV_FILE_VARIABLE}={value} does not name a file.")
    return str(path.resolve())


def _package_directory() -> Path:
    """The installed ``cognee`` package directory (this module lives in ``cognee/shared``)."""
    return Path(__file__).resolve().parent.parent


def _working_directory() -> Path | None:
    try:
        return Path.cwd()
    except OSError:  # the working directory was deleted under the process
        return None


def resolve_env_file() -> str | None:
    """The ``.env`` cognee would load, without loading it."""
    explicit = _explicit_env_file()
    if explicit is not None:
        return explicit
    cwd = _working_directory()
    if cwd is not None:
        found = _find_upward(cwd)
        if found is not None:
            return found
    return _find_upward(_package_directory())


def load_env_file() -> str | None:
    """Load the resolved ``.env`` once, the file winning over preset variables.

    Returns the path that was loaded, or None when there was nothing to load.
    Later calls return the same answer without touching the environment again.
    """
    global _loaded, _resolved
    if not _loaded:
        _resolved = resolve_env_file()
        if _resolved is not None:
            dotenv.load_dotenv(_resolved, override=True)
        _loaded = True
    return _resolved


def describe_resolution(path: str | None) -> str:
    """One log line saying which file was loaded, or that none was."""
    if path:
        return f"Loaded settings from {path}; its values take precedence over preset environment variables."
    return "No .env file found; using process environment variables only."
