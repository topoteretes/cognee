"""Import probe run inside a fresh virtualenv that has ``cognee[<extra>]`` installed.

Prints one JSON object on the last line of stdout::

    {"extra": "neo4j", "ok": true, "imported": [...], "failed": {...}, ...}

Three things are checked, in order:

1. ``import cognee`` still works with the extra installed (a version clash
   introduced by the extra shows up here first).
2. Every top-level module shipped by the extra's *direct* requirements imports.
   The dist -> module mapping comes from ``importlib.metadata`` in the venv, so
   no hand-maintained table can drift from ``pyproject.toml``.
3. The cognee modules the extra is meant to enable import (curated list passed
   in by the test), e.g. the Neo4j adapter for ``neo4j``.

Standalone on purpose: no imports from the test package.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import traceback

# Top-level names some dists expose that are not meant to be imported directly.
_JUNK_TOP_LEVELS = {"tests", "test", "docs", "examples", "example", "scripts", "benchmarks", "bin"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dist_name(requirement: str) -> str:
    """'unstructured[csv, pdf]>=0.18 ; python_version < "3.13"' -> 'unstructured'."""
    head = requirement.split(";", 1)[0].strip()
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", head)
    return match.group(1) if match else head


def _top_levels_for(dist_names: list[str]) -> dict[str, list[str]]:
    from importlib.metadata import packages_distributions

    wanted = {_normalize(d) for d in dist_names}
    by_dist: dict[str, set[str]] = {}
    for module_name, dists in packages_distributions().items():
        if module_name.startswith("_") or module_name in _JUNK_TOP_LEVELS or "." in module_name:
            continue
        for dist in dists:
            if _normalize(dist) in wanted:
                by_dist.setdefault(_normalize(dist), set()).add(module_name)
    return {dist: sorted(mods) for dist, mods in by_dist.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", required=True)
    parser.add_argument("--requirements", default="[]", help="JSON list of requirement strings")
    parser.add_argument("--cognee-modules", default="[]", help="JSON list of cognee module paths")
    args = parser.parse_args()

    requirements = json.loads(args.requirements)
    cognee_modules = json.loads(args.cognee_modules)
    report: dict = {
        "extra": args.extra,
        "python": sys.version.split()[0],
        "imported": [],
        "failed": {},
        "missing_dists": [],
    }

    def try_import(name: str, bucket: str) -> None:
        try:
            importlib.import_module(name)
            report["imported"].append(name)
        except BaseException as error:  # noqa: BLE001 - report everything, even SystemExit
            report["failed"][name] = {
                "bucket": bucket,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-2000:],
            }

    # 1. cognee itself must still import.
    try_import("cognee", "base")

    # 2. every top-level module of the extra's direct requirements.
    dist_names = [_dist_name(r) for r in requirements]
    top_levels = _top_levels_for(dist_names)
    for dist in dist_names:
        key = _normalize(dist)
        if key not in top_levels:
            report["missing_dists"].append(dist)
            continue
        for module_name in top_levels[key]:
            try_import(module_name, f"dependency:{dist}")

    # 3. cognee modules that the extra enables.
    for module_name in cognee_modules:
        try_import(module_name, "cognee")

    report["ok"] = not report["failed"] and not report["missing_dists"]
    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
