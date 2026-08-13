#!/usr/bin/env python3
"""Print the extension-repo version dirs the current pyproject.toml supports.

The ladybug requirement in pyproject.toml is the source of truth for which
ladybug versions cognee supports. This script intersects that constraint with
the verified package-version -> extension-repo-version mapping in
``cognee_db_workers/_kuzu_helpers.py`` and prints the distinct extension
directories to bundle, one per line (e.g. ``v0.18.1``).

Used by scripts/fetch_ladybug_json_extension.sh (release builds) and
cross-checked by the guard tests in
cognee/tests/unit/infrastructure/databases/graph/test_bundled_json_extension.py.

Stdlib-only on purpose: release runners call it before any environment sync.
Fails loudly on constraint syntax it does not understand rather than guessing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))
from cognee_db_workers._kuzu_helpers import _EXTENSION_REPO_VERSIONS  # noqa: E402


def ladybug_requirement(pyproject_text: str) -> str:
    """The ladybug dependency string, without quotes or environment marker."""
    matches = re.findall(r'"(ladybug[^"]*)"', pyproject_text)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one ladybug dependency in pyproject.toml, found {len(matches)}"
        )
    return matches[0].split(";")[0].strip()


def _version_tuple(version: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(\.\d+)*", version):
        raise SystemExit(
            f"cannot compare non-numeric version {version!r} — "
            "extend scripts/ladybug_extension_versions.py"
        )
    return tuple(int(part) for part in version.split("."))


def satisfies(version: str, requirement: str) -> bool:
    """True when *version* satisfies the requirement's specifier list.

    Supports the operators actually used in cognee's pyproject (==, !=, <=,
    >=, <, >) on plain X.Y.Z versions; anything else aborts loudly.
    """
    spec = requirement.removeprefix("ladybug").strip()
    if not spec:
        return True
    have = _version_tuple(version)
    for clause in spec.split(","):
        clause = clause.strip()
        match = re.fullmatch(r"(==|!=|<=|>=|<|>)\s*([\d.]+)", clause)
        if not match:
            raise SystemExit(
                f"unsupported specifier clause {clause!r} — "
                "extend scripts/ladybug_extension_versions.py"
            )
        op, bound = match.group(1), _version_tuple(match.group(2))
        ok = {
            "==": have == bound,
            "!=": have != bound,
            "<=": have <= bound,
            ">=": have >= bound,
            "<": have < bound,
            ">": have > bound,
        }[op]
        if not ok:
            return False
    return True


def supported_extension_dirs() -> list[str]:
    requirement = ladybug_requirement((REPO_ROOT / "pyproject.toml").read_text())
    dirs = sorted(
        {
            ext_dir
            for package_version, ext_dir in _EXTENSION_REPO_VERSIONS.items()
            if satisfies(package_version, requirement)
        }
    )
    if not dirs:
        raise SystemExit(
            f"no _EXTENSION_REPO_VERSIONS entry satisfies {requirement!r} — "
            "extend the mapping in cognee_db_workers/_kuzu_helpers.py "
            "(run `INSTALL JSON;` offline and read the version out of the error URL)"
        )
    return dirs


if __name__ == "__main__":
    print("\n".join(supported_extension_dirs()))
