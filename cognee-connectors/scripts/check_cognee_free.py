#!/usr/bin/env python3
"""Prove that a connector package imports with cognee absent.

Puts ``None`` in ``sys.modules["cognee"]`` — the standard way to make ``import cognee``
(and ``import cognee.anything``) raise ``ModuleNotFoundError`` whether or not cognee is
installed — then imports the package and every module under it, except the
``cognee_host`` subpackage, which is the one place allowed to need cognee
(docs/ARCHITECTURE.md, rule 1). Stops at the first module that fails and names it.

Usage:
    python scripts/check_cognee_free.py cognee_connector_notion
    PYTHONPATH=core:connectors/notion python scripts/check_cognee_free.py cognee_connector_notion
    python scripts/check_cognee_free.py cognee_connector_slack --allow cognee_host --allow other_host

Exit status: 0 when every module imported; 1 when one failed (the message names the module
and, when the cause was cognee, says so); 2 on a usage error.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback
from types import ModuleType

DEFAULT_ALLOWED = ("cognee_host",)


def _is_cognee_import_error(error: BaseException) -> bool:
    name = getattr(error, "name", None) or ""
    return isinstance(error, ImportError) and (name == "cognee" or name.startswith("cognee."))


def _import(name: str) -> ModuleType | None:
    """Import ``name``; on failure print why and return ``None``."""
    try:
        return importlib.import_module(name)
    except BaseException as error:  # noqa: BLE001 - every failure is reported, not hidden
        if _is_cognee_import_error(error):
            print(f"FAIL: {name} pulled cognee in: {error}")
        else:
            print(f"FAIL: {name} did not import ({type(error).__name__}: {error})")
        print("".join(traceback.format_exception(error)).rstrip())
        return None


def walk(package: ModuleType, allowed: set[str], imported: list[str]) -> str | None:
    """Import every module under ``package`` depth-first; the first failing name, or ``None``."""
    for info in sorted(
        pkgutil.iter_modules(package.__path__, prefix=package.__name__ + "."),
        key=lambda item: item.name,
    ):
        if info.name.rsplit(".", 1)[-1] in allowed:
            print(f"skip: {info.name} (may import cognee)")
            continue
        module = _import(info.name)
        if module is None:
            return info.name
        imported.append(info.name)
        if info.ispkg and hasattr(module, "__path__"):
            failed = walk(module, allowed, imported)
            if failed:
                return failed
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a package and all of its modules with cognee made unimportable."
    )
    parser.add_argument("module", help="the import name, e.g. cognee_connector_notion")
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "a subpackage or module name that may import cognee and is skipped "
            f"(default: {', '.join(DEFAULT_ALLOWED)}; repeatable)"
        ),
    )
    args = parser.parse_args(argv)
    allowed = set(DEFAULT_ALLOWED) | set(args.allow)

    # Any `import cognee` from here on raises ModuleNotFoundError, installed or not.
    sys.modules["cognee"] = None  # type: ignore[assignment]

    imported: list[str] = []
    package = _import(args.module)
    if package is None:
        return 1
    imported.append(args.module)
    if hasattr(package, "__path__"):
        failed = walk(package, allowed, imported)
        if failed:
            print(
                f"\n{failed} needs cognee or does not import; see above. "
                f"({len(imported)} module(s) imported before it.)"
            )
            return 1
    print(f"OK: {len(imported)} module(s) of {args.module} import without cognee.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
