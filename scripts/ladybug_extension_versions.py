#!/usr/bin/env python3
"""Filter extension-repo version dirs down to what pyproject.toml supports.

Reads candidate directory names (``v0.18.1`` style, one per line, as listed in
the ladybug extension repo) on stdin and prints the ones cognee should bundle,
based on the ladybug requirement in pyproject.toml — the source of truth for
the supported version range. Non-version entries (``vdev``, ``dataset``) are
skipped.

One subtlety: an extension dir can *trail* the package versions it serves
(ladybug 0.18.2 requests ``v0.18.1``), so when the range floor itself has no
exact dir, the newest dir below the floor is included too — it is the one
serving the floor version. At runtime the engine announces its exact dir via
the probe in ``cognee_db_workers/_kuzu_helpers.py``; this filter only decides
what to ship.

Stdlib-only on purpose: release runners call it before any environment sync.
Fails loudly on constraint syntax it does not understand rather than guessing.

Usage: docker run --rm --entrypoint ls <extension-repo-image> \
           /usr/share/nginx/html | python3 scripts/ladybug_extension_versions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_VERSION_DIR = re.compile(r"v\d+(\.\d+)*$")


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


def _clauses(requirement: str) -> list[tuple[str, tuple[int, ...]]]:
    spec = requirement.removeprefix("ladybug").strip()
    if not spec:
        return []
    clauses = []
    for clause in spec.split(","):
        clause = clause.strip()
        match = re.fullmatch(r"(==|!=|<=|>=|<|>)\s*([\d.]+)", clause)
        if not match:
            raise SystemExit(
                f"unsupported specifier clause {clause!r} — "
                "extend scripts/ladybug_extension_versions.py"
            )
        clauses.append((match.group(1), _version_tuple(match.group(2))))
    return clauses


def satisfies(version: str, requirement: str) -> bool:
    """True when *version* satisfies every specifier clause (PEP 440 subset)."""
    have = _version_tuple(version)
    checks = {
        "==": lambda bound: have == bound,
        "!=": lambda bound: have != bound,
        "<=": lambda bound: have <= bound,
        ">=": lambda bound: have >= bound,
        "<": lambda bound: have < bound,
        ">": lambda bound: have > bound,
    }
    return all(checks[op](bound) for op, bound in _clauses(requirement))


def supported_extension_dirs(candidates: list[str], requirement: str) -> list[str]:
    """The candidate dirs to bundle for the requirement's version range."""
    versioned = sorted(
        (d for d in candidates if _VERSION_DIR.fullmatch(d)),
        key=lambda d: _version_tuple(d[1:]),
    )
    in_range = [d for d in versioned if satisfies(d[1:], requirement)]

    # Upper-bound clauses only — a dir failing these serves nothing we support.
    uppers = [(op, bound) for op, bound in _clauses(requirement) if op in ("<=", "<")]
    below_floor = [
        d
        for d in versioned
        if d not in in_range
        and all(
            {"<=": _version_tuple(d[1:]) <= b, "<": _version_tuple(d[1:]) < b}[op]
            for op, b in uppers
        )
    ]
    # The floor version's dir can trail below the floor (dirs lag package
    # versions); when no dir matches the floor exactly, ship the newest one
    # below it.
    floors = [bound for op, bound in _clauses(requirement) if op == ">="]
    floor_has_exact_dir = any(_version_tuple(d[1:]) in floors for d in in_range)
    if below_floor and floors and not floor_has_exact_dir:
        in_range.insert(0, below_floor[-1])

    if not in_range:
        raise SystemExit(
            f"no candidate extension dir satisfies {requirement!r} — candidates were {candidates!r}"
        )
    return in_range


if __name__ == "__main__":
    if sys.stdin.isatty():
        raise SystemExit(__doc__)
    requirement = ladybug_requirement((REPO_ROOT / "pyproject.toml").read_text())
    candidates = [line.strip() for line in sys.stdin if line.strip()]
    print("\n".join(supported_extension_dirs(candidates, requirement)))
