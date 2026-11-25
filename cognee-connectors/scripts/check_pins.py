#!/usr/bin/env python3
"""Check the dependency pins of every connector package, and the shared rules on core.

Policy (docs/ARCHITECTURE.md, rules 1 and 7):

  - every ``connectors/*/pyproject.toml`` depends on ``cognee-connectors-core`` with a lower
    bound and the upper bound ``<1`` (connectors pin ``cognee-connectors-core>=0.1,<1``
    until core reaches 1.0);
  - no package has a runtime dependency on cognee: ``cognee`` never appears in
    ``[project.dependencies]``;
  - cognee is allowed only in the ``cognee`` extra (``[project.optional-dependencies].cognee``)
    and must carry a ``<2`` upper bound there;
  - the build backend is pinned, ``[build-system] requires = ["hatchling==1.32.0"]``: the backend
    decides the wheel's Metadata-Version, which the PyPI publish action validates with a pinned
    twine, so it moves together with that action's pin in .github/workflows/release_connectors.yml;
  - the version is PEP 440.

Handles the requirement spellings a pyproject can use: extras
(``cognee-connectors-core[dlt]>=0.1,<1``), markers (``cognee>=1.6.5,<2 ; python_version < '3.14'``)
and URL/git requirements (``cognee @ git+https://...``, flagged as not publishable).

Usage:
    python scripts/check_pins.py              # connectors/*/pyproject.toml and core/pyproject.toml
    python scripts/check_pins.py PATH [PATH]  # the given pyproject.toml files
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from packages import REPO_ROOT, TomlError, load_toml, normalize_name

CORE_NAME = "cognee-connectors-core"
REQUIRED_BUILD_REQUIRES = ["hatchling==1.32.0"]
REQUIRED_BUILD_BACKEND = "hatchling.build"

# PEP 440, Appendix B.
_PEP440 = re.compile(
    r"^\s*v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?[0-9]*)?"
    r"(?:-[0-9]+|[-_.]?(?:post|rev|r)[-_.]?[0-9]*)?"
    r"(?:[-_.]?dev[-_.]?[0-9]*)?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?\s*$",
    re.IGNORECASE,
)
_HEAD = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*"
    r"(?:\[(?P<extras>[^\]]*)\])?\s*(?P<rest>.*?)\s*$"
)
_SPECIFIER = re.compile(r"^(===|==|!=|<=|>=|<|>|~=)\s*(\S+)$")


@dataclass
class Requirement:
    """One PEP 508 requirement, reduced to what the policy needs."""

    raw: str
    name: str
    url: str | None = None
    specifiers: list[tuple[str, str]] = field(default_factory=list)
    marker: str | None = None

    def has_lower_bound(self) -> bool:
        return any(op in (">=", ">", "==", "===", "~=") for op, _ in self.specifiers)

    def has_upper_bound_at_most(self, major: int) -> bool:
        """A strict ``<`` bound at or below ``major``.0 (``<1``, ``<1.0``, ``<0.9`` for 1)."""
        for op, version in self.specifiers:
            if op != "<":
                continue
            release = _release_tuple(version)
            if release is not None and release <= (major,):
                return True
        return False


def _release_tuple(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^v?(?:[0-9]+!)?([0-9]+(?:\.[0-9]+)*)", version.strip())
    if not match:
        return None
    parts = [int(part) for part in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def parse_requirement(raw: str) -> Requirement:
    match = _HEAD.match(raw)
    if not match:
        raise ValueError(f"cannot parse requirement {raw!r}")
    rest = match.group("rest")
    marker = None
    if ";" in rest:
        rest, marker = (part.strip() for part in rest.split(";", 1))
    requirement = Requirement(raw=raw, name=normalize_name(match.group("name")), marker=marker)
    if rest.startswith("@"):
        requirement.url = rest[1:].strip()
        return requirement
    for part in rest.strip("()").split(","):
        part = part.strip()
        if not part:
            continue
        specifier = _SPECIFIER.match(part)
        if not specifier:
            raise ValueError(f"cannot parse specifier {part!r} in requirement {raw!r}")
        requirement.specifiers.append((specifier.group(1), specifier.group(2)))
    return requirement


def check_pyproject(path: Path, is_connector: bool) -> list[str]:
    """The policy violations of one pyproject.toml, as messages naming the file."""
    label = path.parent.name
    try:
        config = load_toml(path)
    except (TomlError, OSError) as error:
        return [f"{label}: cannot read {path}: {error}"]

    errors: list[str] = []
    project = config.get("project")
    if not isinstance(project, dict):
        return [f"{label}: {path} has no [project] table"]

    name = project.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{label}: [project] has no name")
    version = project.get("version")
    if not isinstance(version, str) or not _PEP440.match(version):
        errors.append(f"{label}: version {version!r} is not a valid PEP 440 version")

    build_system = config.get("build-system", {})
    if build_system.get("requires") != REQUIRED_BUILD_REQUIRES:
        errors.append(
            f"{label}: [build-system] requires must be {REQUIRED_BUILD_REQUIRES!r} "
            f"(found {build_system.get('requires')!r})"
        )
    if build_system.get("build-backend") != REQUIRED_BUILD_BACKEND:
        errors.append(
            f"{label}: [build-system] build-backend must be {REQUIRED_BUILD_BACKEND!r} "
            f"(found {build_system.get('build-backend')!r})"
        )

    dependencies: list[Requirement] = []
    for raw in project.get("dependencies", []):
        try:
            dependencies.append(parse_requirement(str(raw)))
        except ValueError as error:
            errors.append(f"{label}: {error}")

    for requirement in dependencies:
        if requirement.name == "cognee":
            errors.append(
                f"{label}: cognee is a runtime dependency ({requirement.raw!r}); it is only "
                "allowed in the `cognee` extra"
            )

    if is_connector:
        core = [requirement for requirement in dependencies if requirement.name == CORE_NAME]
        if not core:
            errors.append(f"{label}: [project.dependencies] does not list {CORE_NAME}")
        for requirement in core:
            if requirement.url:
                errors.append(
                    f"{label}: {CORE_NAME} is a URL/git dependency (not publishable). "
                    f"Found: {requirement.raw!r}"
                )
                continue
            if not requirement.has_lower_bound():
                errors.append(
                    f"{label}: {CORE_NAME} dependency is missing a lower bound (>=). "
                    f"Found: {requirement.raw!r}"
                )
            if not requirement.has_upper_bound_at_most(1):
                errors.append(
                    f"{label}: {CORE_NAME} dependency is missing the upper bound <1. "
                    f"Found: {requirement.raw!r}"
                )

    for extra, raws in project.get("optional-dependencies", {}).items():
        for raw in raws:
            try:
                requirement = parse_requirement(str(raw))
            except ValueError as error:
                errors.append(f"{label}: {error}")
                continue
            if requirement.name != "cognee":
                continue
            if extra != "cognee":
                errors.append(
                    f"{label}: cognee appears in the {extra!r} extra; it is only allowed in "
                    f"the `cognee` extra. Found: {requirement.raw!r}"
                )
                continue
            if requirement.url:
                errors.append(
                    f"{label}: cognee uses a URL/git dependency (not publishable). "
                    f"Found: {requirement.raw!r}"
                )
            elif not requirement.has_upper_bound_at_most(2):
                errors.append(
                    f"{label}: cognee in the `cognee` extra is missing the upper bound <2. "
                    f"Found: {requirement.raw!r}"
                )
    return errors


def default_targets(root: Path) -> list[tuple[Path, bool]]:
    targets: list[tuple[Path, bool]] = []
    core = root / "core" / "pyproject.toml"
    if core.is_file():
        targets.append((core, False))
    connectors = root / "connectors"
    if connectors.is_dir():
        for directory in sorted(connectors.iterdir()):
            pyproject = directory / "pyproject.toml"
            if directory.is_dir() and pyproject.is_file():
                targets.append((pyproject, True))
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the dependency pins of the workspace packages."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="pyproject.toml files to check (default: core/ and every connectors/*)",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.paths:
        targets = [(path, "connectors" in path.resolve().parts) for path in args.paths]
    else:
        targets = default_targets(args.root)

    all_errors: list[str] = []
    for path, is_connector in targets:
        all_errors.extend(check_pyproject(path, is_connector))

    connectors = sum(1 for _, is_connector in targets if is_connector)
    print(
        f"Checked {len(targets)} package(s): {connectors} connector(s), {len(targets) - connectors} core."
    )
    if not targets:
        print("No packages with a pyproject.toml found.")
        return 0

    if all_errors:
        print(f"\n{len(all_errors)} pinning violation(s) found:\n")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    print(
        "All packages pin cognee-connectors-core, keep cognee out of their runtime "
        "dependencies and pin the build backend."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
