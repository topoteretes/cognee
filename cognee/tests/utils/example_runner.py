"""Helpers for loading and running ``examples/`` scripts inside tests.

``import_example`` loads an example module by file path (so the on-disk script
is exercised verbatim -- no copy-pasting example logic into tests) and returns
the imported module, from which a test typically awaits ``invoke_example_main``.

``requires_docker`` / ``requires_aws`` are skip markers for the minority of
examples that genuinely need an external service; they keep the default suite
runnable with zero secrets and no daemon.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Mapping

import pytest

# repo root: cognee/tests/utils/example_runner.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"


def example_path(rel_path: str) -> Path:
    """Resolve a path relative to the repo root (accepts ``examples/...``)."""
    path = (REPO_ROOT / rel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Example script not found: {path}")
    return path


@contextmanager
def _script_import_context(path: Path):
    """Isolate import side effects: ``sys.argv``, CWD, and ``sys.path``."""
    script_dir = str(path.parent)
    old_argv = sys.argv
    old_cwd = os.getcwd()
    added_to_path = script_dir not in sys.path
    if added_to_path:
        sys.path.insert(0, script_dir)
    sys.argv = [str(path)]
    os.chdir(script_dir)
    try:
        yield
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        if added_to_path and script_dir in sys.path:
            sys.path.remove(script_dir)


def import_example(
    rel_path: str,
    *,
    pre_import_env: Mapping[str, str] | None = None,
    module_name: str | None = None,
) -> ModuleType:
    """Import an example script by path and return the module object.

    Parameters
    ----------
    rel_path:
        Path to the script relative to the repo root, e.g.
        ``"examples/guides/recall_core.py"``.
    pre_import_env:
        Environment variables to set *before* the module body executes. Some
        examples read/mutate ``os.environ`` at import time (e.g. the relational
        migration demos hardcode ``MIGRATION_DB_PROVIDER`` at line 8), so the
        override must be in place before ``exec_module`` runs.
    module_name:
        Optional explicit module name; defaults to a unique name derived from
        the path to avoid clobbering ``sys.modules``.
    """
    path = example_path(rel_path)

    if pre_import_env:
        for key, value in pre_import_env.items():
            os.environ[key] = value

    name = module_name or "cognee_example_" + rel_path.replace("/", "_").replace(".py", "")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    with _script_import_context(path):
        spec.loader.exec_module(module)
    return module


async def invoke_example_main(
    module: ModuleType,
    rel_path: str,
    *,
    work_dir: Path | None = None,
    chdir_to_script: bool = False,
) -> None:
    """Await ``module.main()`` with pytest-safe ``sys.argv`` and CWD.

    Parameters
    ----------
    work_dir:
        Directory to use as CWD while ``main()`` runs (typically the test
        ``tmp_path`` sandbox). Defaults to the script directory when
        ``chdir_to_script`` is True, otherwise the current working directory.
    chdir_to_script:
        When True, run ``main()`` with CWD set to the script's parent so
        examples that read CWD-relative files (e.g. ``prompts/prompt1.txt``)
        work without per-test workarounds.
    """
    path = example_path(rel_path)
    old_argv = sys.argv
    old_cwd = os.getcwd()
    if chdir_to_script:
        target_dir = path.parent
    elif work_dir is not None:
        target_dir = work_dir
    else:
        target_dir = Path(old_cwd)
    sys.argv = [str(path)]
    os.chdir(target_dir)
    try:
        await module.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _aws_credentials_available() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))


def requires_docker(reason: str = "requires a running Docker daemon"):
    """Skip marker for tests that need Docker (e.g. Testcontainers backends)."""
    return pytest.mark.skipif(not _docker_available(), reason=reason)


def requires_aws(reason: str = "requires AWS credentials"):
    """Skip marker for tests that need real AWS services (S3, Neptune)."""
    return pytest.mark.skipif(not _aws_credentials_available(), reason=reason)
