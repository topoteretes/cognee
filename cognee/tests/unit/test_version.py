"""get_cognee_version() on the ways cognee can be laid out on disk.

Each case copies ``cognee/version.py`` into a bare package and calls it in a
separate interpreter started with ``-I -S``: no site-packages, so the test
environment's own cognee install cannot answer for the copy.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import cognee.version

STAMP = '__version__ = VERSION = "9.8.7"\n'


def _package(root: Path, *, stamp: str | None = None) -> Path:
    package = root / "cognee"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    shutil.copy(cognee.version.__file__, package / "version.py")
    if stamp is not None:
        (package / "_version.py").write_text(stamp)
    return root


def _version(root: Path) -> str:
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from cognee.version import get_cognee_version; print(get_cognee_version())"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_package_only_copy_reports_the_stamped_release(tmp_path):
    assert _version(_package(tmp_path, stamp=STAMP)) == "9.8.7"


def test_package_only_copy_without_a_stamp_is_unknown(tmp_path):
    assert _version(_package(tmp_path)) == "unknown"


def test_source_checkout_ignores_a_stale_stamp(tmp_path):
    root = _package(tmp_path, stamp=STAMP)
    (root / "pyproject.toml").write_text('[project]\nname = "cognee"\nversion = "1.2.3"\n')
    assert _version(root) == "1.2.3-local"


def test_installed_metadata_wins_over_the_stamp(tmp_path):
    root = _package(tmp_path, stamp=STAMP)
    dist_info = root / "cognee-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: cognee\nVersion: 1.2.3\n")
    assert _version(root) == "1.2.3"
