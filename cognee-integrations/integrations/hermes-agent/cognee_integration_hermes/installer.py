"""Materialize the Hermes directory plugin from the installed pip package.

Hermes discovers memory providers through pip entry points and by scanning
``$HERMES_HOME/plugins/``. The ``cognee-hermes-install`` console script adds
the directory layout needed for the CLI and dashboard, copied from the wheel.
The full flow:

    pip install cognee-integration-hermes-agent
    cognee-hermes-install
    hermes memory setup

The one file the scanner actually keys on is the plugin root's ``__init__.py``:
Hermes' ``plugins.memory._is_memory_provider_dir`` skips — silently — any
directory without one (it must mention ``MemoryProvider`` or
``register_memory_provider`` in its first 8KB). Versions 1.0.0–1.2.0 of this
installer never laid that file down, so a pip install produced a plugin Hermes
ignored while everything reported success (issue #382). It ships in the wheel
as ``_plugin_root/plugin_init.py`` — naming it ``__init__.py`` in place would
turn ``_plugin_root`` into an importable subpackage and change how setuptools
treats its data files — and is written to the target under its real name.

Because Hermes runs the *copy*, ``pip install -U`` alone changes nothing —
re-run ``cognee-hermes-install`` after upgrading. ``hermes cognee status``
warns when the pip package is newer than the installed copy.

The plugin-root files ship inside the package under ``_plugin_root/`` (a wheel
cannot carry files outside its packages); a test pins them byte-identical to
the repository's canonical copies.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .catalog import catalog_name
from .config import resolve_hermes_home

# Packaged name -> name at the plugin root. plugin_init.py becomes the
# __init__.py that makes the directory discoverable as a memory provider.
_ROOT_FILES = {
    "plugin.yaml": "plugin.yaml",
    "cli.py": "cli.py",
    "after-install.md": "after-install.md",
    "plugin_init.py": "__init__.py",
}


def install(hermes_home: str | Path | None = None) -> Path:
    """Copy the plugin into ``$HERMES_HOME/plugins/cognee``; return that path."""
    home = resolve_hermes_home(hermes_home)
    if home is None:
        # No Hermes on the path to ask; the documented default.
        home = Path.home() / ".hermes"
    package_src = Path(__file__).resolve().parent
    root_src = package_src / "_plugin_root"
    missing = [name for name in _ROOT_FILES if not (root_src / name).is_file()]
    if missing:
        raise RuntimeError(
            "packaged plugin files missing (%s) — this looks like a broken or "
            "source-tree install; reinstall the package from PyPI" % ", ".join(missing)
        )

    target = home / "plugins" / "cognee"
    name = catalog_name(target)
    if name:
        raise RuntimeError(
            f"Refusing to overwrite catalog-managed plugin at {target}. "
            f"Run `hermes plugins update {name}` instead."
        )
    target.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in _ROOT_FILES.items():
        shutil.copyfile(root_src / source_name, target / target_name)

    # Replace the package wholesale so removed modules never linger.
    dest_package = target / package_src.name
    if dest_package.exists():
        shutil.rmtree(dest_package)
    shutil.copytree(
        package_src,
        dest_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cognee-hermes-install",
        description="Install the Cognee memory plugin into a Hermes Agent home.",
    )
    parser.add_argument(
        "--home",
        default=None,
        help="HERMES_HOME to install into (default: auto-detect, else ~/.hermes)",
    )
    args = parser.parse_args(argv)
    try:
        target = install(args.home)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Cognee memory plugin installed at {target}")
    print("Next: run `hermes memory setup cognee` to make it your memory provider.")
    print("After a `pip install -U`, re-run `cognee-hermes-install` to update this copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
