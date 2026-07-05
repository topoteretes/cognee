"""Regression test: pylance must not be in core runtime dependencies.

pylance provides the `lance` module used by lancedb, but it is not
imported directly by cognee code. It should be in optional/dev deps,
not in the core [project].dependencies list (issue #3827).
"""

from __future__ import annotations

import pathlib

import pytest
import tomllib


@pytest.fixture
def pyproject_data():
    pyproject_path = pathlib.Path(__file__).resolve().parents[3] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def test_pylance_not_in_runtime_dependencies(pyproject_data):
    """pylance must not appear in [project].dependencies."""
    runtime_deps = pyproject_data.get("project", {}).get("dependencies", [])
    pylance_deps = [d for d in runtime_deps if d.strip().lower().startswith("pylance")]
    assert pylance_deps == [], (
        f"pylance should not be in [project].dependencies (runtime). "
        f"Found: {pylance_deps}. Move it to [project.optional-dependencies.dev]."
    )


def test_pylance_in_dev_dependencies(pyproject_data):
    """pylance should be available in the dev optional dependencies."""
    dev_deps = pyproject_data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    pylance_deps = [d for d in dev_deps if d.strip().lower().startswith("pylance")]
    assert len(pylance_deps) >= 1, (
        "pylance should be listed in [project.optional-dependencies.dev] "
        "so it is available for development/testing with lancedb."
    )
