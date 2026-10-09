"""The unstructured loader must not import unstructured when cognee is imported.

Importing unstructured probes libmagic through python-magic. On the Windows CI
runners that probe wedged forever at import time, so every ``import cognee``
hung (SDK-533). The loader therefore only checks that unstructured is installed
when it registers, and imports it on the first ``load()`` call.

Each check runs in a subprocess so ``sys.modules`` starts clean.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]


def _run(code: str, *, pythonpath: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "COGNEE_SKIP_PREFLIGHT": "true"}
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
    )


def _write_unstructured_stub(root: Path) -> None:
    """A minimal ``unstructured`` package: enough for the loader to register and load."""
    pkg = root / "unstructured" / "partition"
    pkg.mkdir(parents=True)
    (root / "unstructured" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "auto.py").write_text(
        "def partition(**kwargs):\n    return ['first element', ' ', 'second element']\n"
    )


def test_importing_loaders_does_not_import_unstructured_or_magic():
    result = _run(
        """
        import sys
        import cognee.infrastructure.loaders.supported_loaders  # noqa: F401
        leaked = sorted(m for m in sys.modules if m == "magic" or m.split(".")[0] in ("unstructured", "magic"))
        assert not leaked, leaked
        """
    )
    assert result.returncode == 0, result.stderr


def test_loader_registers_and_imports_unstructured_only_on_load(tmp_path):
    _write_unstructured_stub(tmp_path)
    document = tmp_path / "sample.docx"
    document.write_bytes(b"not really a docx")
    result = _run(
        f"""
        import asyncio, sys
        from cognee.infrastructure.loaders.supported_loaders import supported_loaders
        assert "unstructured_loader" in supported_loaders, sorted(supported_loaders)
        assert "unstructured.partition.auto" not in sys.modules

        loader = supported_loaders["unstructured_loader"]()
        text = asyncio.run(loader.load({str(document)!r}, persist=False))
        assert text == "first element\\n\\nsecond element", repr(text)
        assert "unstructured.partition.auto" in sys.modules
        """,
        pythonpath=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_loader_is_not_registered_when_unstructured_is_missing():
    result = _run(
        """
        import sys
        sys.modules["unstructured"] = None  # what an uninstalled package looks like to find_spec
        from cognee.infrastructure.loaders.supported_loaders import supported_loaders
        assert "unstructured_loader" not in supported_loaders, sorted(supported_loaders)
        try:
            import cognee.infrastructure.loaders.external.unstructured_loader  # noqa: F401
        except ImportError as exc:
            assert "pip install unstructured" in str(exc), exc
        else:
            raise AssertionError("expected ImportError")
        """
    )
    assert result.returncode == 0, result.stderr
