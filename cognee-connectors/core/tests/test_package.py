"""The package surface: version, exports, and importing with cognee absent."""

import subprocess
import sys
from pathlib import Path

import cognee_connectors_core

_MODULES = [
    "cognee_connectors_core",
    "cognee_connectors_core.base",
    "cognee_connectors_core.descriptor",
    "cognee_connectors_core.dlt_conventions",
    "cognee_connectors_core.errors",
    "cognee_connectors_core.http",
    "cognee_connectors_core.oauth2",
    "cognee_connectors_core.oauth_state",
    "cognee_connectors_core.registry",
    "cognee_connectors_core.settings",
    "cognee_connectors_core.signatures",
    "cognee_connectors_core.testing",
]


def test_version():
    assert cognee_connectors_core.__version__ == "0.1.0"


def test_every_exported_name_resolves():
    for name in cognee_connectors_core.__all__:
        assert hasattr(cognee_connectors_core, name), name


def test_the_package_is_typed():
    assert (Path(cognee_connectors_core.__file__).parent / "py.typed").exists()


def test_every_module_imports_with_cognee_absent():
    """The one rule the whole repo rests on: no runtime cognee dependency.

    Run in a subprocess with ``cognee`` blocked in ``sys.modules`` (a ``None``
    entry makes any import of it raise), so a stray ``import cognee`` in any
    module fails here rather than on a user's machine without cognee.
    """
    package_root = Path(cognee_connectors_core.__file__).parent.parent
    script = (
        "import sys; sys.modules['cognee'] = None; "
        + "; ".join(f"import {module}" for module in _MODULES)
        + "; print('cognee-free ok')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(package_root),
        env={"PYTHONPATH": str(package_root), "PATH": ""},
    )

    assert completed.returncode == 0, completed.stderr
    assert "cognee-free ok" in completed.stdout
