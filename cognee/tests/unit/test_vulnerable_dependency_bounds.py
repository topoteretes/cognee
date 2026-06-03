"""Regression guards for previously-remediated vulnerable dependencies.

These tests fail if a future change reintroduces a known-vulnerable lower bound
or pin. They parse ``pyproject.toml`` directly so they do not depend on the deps
being installed.

Covered advisories:
- cbor2 < 5.8.0 -> CVE-2025-68131 (issue #1950)
- fastapi-users < 15 pinning python-multipart 0.0.20 (issue #2101)
- diskcache pickle GHSA-w8v5-vhqr-4h9v (issue #2957) -> dependency removed
"""

import re
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def _load_dependencies() -> list[str]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            if tomllib is None:  # pragma: no cover
                pytest.skip("tomllib unavailable")
            data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            return list(data.get("project", {}).get("dependencies", []))
    pytest.skip("pyproject.toml not found")


def _find(deps: list[str], name: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(name)}\b", re.IGNORECASE)
    for dep in deps:
        # Strip extras, e.g. "fastapi-users[sqlalchemy]>=15.0.2".
        base = re.sub(r"\[.*?\]", "", dep)
        if pattern.match(base.strip()):
            return dep
    return None


def _min_version(spec: str) -> tuple[int, ...]:
    """Extract the >= lower bound from a PEP 508 spec as a version tuple."""
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", spec)
    assert match, f"expected a >= lower bound in {spec!r}"
    return tuple(int(p) for p in match.group(1).split("."))


def test_cbor2_lower_bound_not_vulnerable():
    """cbor2 must require >= 5.8.0 (CVE-2025-68131, issue #1950)."""
    deps = _load_dependencies()
    spec = _find(deps, "cbor2")
    assert spec is not None, "cbor2 dependency missing"
    assert _min_version(spec) >= (5, 8, 0), spec


def test_fastapi_users_lower_bound_not_vulnerable():
    """fastapi-users must require >= 15 (issue #2101)."""
    deps = _load_dependencies()
    spec = _find(deps, "fastapi-users")
    assert spec is not None, "fastapi-users dependency missing"
    assert _min_version(spec) >= (15,), spec


def test_python_multipart_not_pinned_to_vulnerable_version():
    """python-multipart must not be pinned to the vulnerable 0.0.20 (issue #2101)."""
    deps = _load_dependencies()
    spec = _find(deps, "python-multipart")
    if spec is None:
        # Acceptable: not a direct dependency.
        return
    assert "==0.0.20" not in spec.replace(" ", ""), spec
    assert _min_version(spec) >= (0, 0, 22), spec


def test_diskcache_dependency_removed():
    """diskcache (GHSA-w8v5-vhqr-4h9v) must not be a runtime dependency (issue #2957)."""
    deps = _load_dependencies()
    assert _find(deps, "diskcache") is None, "diskcache must be removed as a runtime dependency"
