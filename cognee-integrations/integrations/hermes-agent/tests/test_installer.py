"""Tests for the pip-install bridge (``cognee-hermes-install``).

Hermes discovers memory providers by directory scan and by the
``hermes_agent.memory_providers`` entry-point group; the installer must
materialize the exact plugin shape the scanner expects — including the root
``__init__.py`` the scanner keys on (issue #382: three releases shipped
without it, so pip installs were silently ignored). The drift tests are the
load-bearing ones — the wheel ships *copies* of the repo-root plugin files,
and a stale copy would publish a plugin.yaml that disagrees with the
repository.

Runs under pytest or standalone (``python3 tests/test_installer.py``); touches
nothing outside temporary directories.
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import installer  # noqa: E402

_PACKAGE = ROOT / "cognee_integration_hermes"


class TestPackagedCopiesMatchTheRepo(unittest.TestCase):
    """The wheel's _plugin_root/* must be byte-identical to the repo root files."""

    def _assert_same(self, name, canonical_name=None):
        packaged = (_PACKAGE / "_plugin_root" / name).read_bytes()
        canonical = (ROOT / (canonical_name or name)).read_bytes()
        self.assertEqual(
            packaged,
            canonical,
            f"{name}: the packaged copy under _plugin_root/ has drifted from the "
            f"repo-root canonical file — re-copy it before publishing",
        )

    def test_plugin_yaml(self):
        self._assert_same("plugin.yaml")

    def test_cli_shim(self):
        self._assert_same("cli.py")

    def test_after_install(self):
        self._assert_same("after-install.md")

    def test_plugin_init(self):
        # Packaged as plugin_init.py (an __init__.py inside _plugin_root/ would
        # make it an importable subpackage); installed as __init__.py.
        self._assert_same("plugin_init.py", canonical_name="__init__.py")

    def test_plugin_yaml_version_matches_pyproject(self):
        # The CHANGELOG contract: one version across plugin.yaml and pyproject.
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        manifest_version = next(
            line.split(":", 1)[1].strip()
            for line in manifest.splitlines()
            if line.startswith("version:")
        )
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'version = "{manifest_version}"', pyproject)


class TestInstall(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def test_lays_down_the_plugin_shape_hermes_scans_for(self):
        target = installer.install(self.home)
        self.assertEqual(target, self.home / "plugins" / "cognee")
        for name in ("plugin.yaml", "cli.py", "after-install.md", "__init__.py"):
            self.assertTrue((target / name).is_file(), f"missing root file {name}")
        # The package itself, importable next to the root files.
        self.assertTrue((target / "cognee_integration_hermes" / "provider.py").is_file())
        self.assertTrue((target / "cognee_integration_hermes" / "exit_watcher.py").is_file())

    def test_installed_dir_passes_hermes_discovery_heuristic(self):
        # Regression for issue #382: 1.0.0–1.2.0 laid down no root __init__.py,
        # and Hermes' plugins.memory._is_memory_provider_dir silently skips any
        # directory without one — every pip install produced a plugin Hermes
        # ignored. This mirrors that heuristic exactly: __init__.py must exist
        # and mention MemoryProvider or register_memory_provider in its first
        # 8KB.
        target = installer.install(self.home)
        init_file = target / "__init__.py"
        self.assertTrue(init_file.exists(), "plugin root has no __init__.py")
        source = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
        self.assertTrue(
            "register_memory_provider" in source or "MemoryProvider" in source,
            "__init__.py would fail Hermes' memory-provider text heuristic",
        )

    def test_pyproject_declares_the_memory_providers_entry_point(self):
        # The other half of issue #382: the memory loader scans the
        # hermes_agent.memory_providers group; 1.0.0–1.2.0 declared only the
        # generic hermes_agent.plugins group, which it never reads. Parsed by
        # section-splitting rather than tomllib, which 3.10 (our floor) lacks.
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        header = '[project.entry-points."hermes_agent.memory_providers"]'
        self.assertIn(header, text)
        section = text.split(header, 1)[1].split("\n[", 1)[0]
        self.assertIn('cognee = "cognee_integration_hermes"', section)

    def test_bytecode_caches_are_not_shipped(self):
        target = installer.install(self.home)
        leftovers = list(target.rglob("__pycache__")) + list(target.rglob("*.pyc"))
        self.assertEqual(leftovers, [])

    def test_reinstall_replaces_the_package_wholesale(self):
        target = installer.install(self.home)
        stale = target / "cognee_integration_hermes" / "removed_in_next_version.py"
        stale.write_text("# stale module from an older release\n", encoding="utf-8")
        installer.install(self.home)
        self.assertFalse(
            stale.exists(),
            "reinstall must replace the package dir, not merge into it — "
            "stale modules from older releases would otherwise linger",
        )

    def test_missing_packaged_files_fail_loudly(self):
        with mock.patch.object(installer, "_ROOT_FILES", {"no-such-file.txt": "no-such-file.txt"}):
            with self.assertRaisesRegex(RuntimeError, "packaged plugin files missing"):
                installer.install(self.home)

    def test_main_honours_home_and_prints_next_steps(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = installer.main(["--home", str(self.home)])
        self.assertEqual(code, 0)
        self.assertTrue((self.home / "plugins" / "cognee" / "plugin.yaml").is_file())
        self.assertIn("hermes memory setup", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
