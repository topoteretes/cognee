"""Guards the ladybug dependency against environment-marker regressions.

The old macOS-gating marker (``platform_release >= '24.'``) could not be
expressed correctly across installer generations: packaging <= 24.x needed the
invalid ``'24.'`` literal to avoid crashing on Linux kernel strings, while
packaging >= 25 (vendored in current pip) dropped the string-ordering fallback,
evaluated the marker False on macOS, and **silently skipped ladybug** — a
pip-installed cognee then failed at ``import cognee``. The dependency is
therefore unconditional; anyone reintroducing a marker must consciously delete
this test and re-verify against every packaging generation.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ladybug_requirement() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    matches = re.findall(r'"(ladybug[^"]*)"', text)
    assert len(matches) == 1, f"expected exactly one ladybug dependency, found {matches}"
    return matches[0]


def test_ladybug_dependency_has_no_environment_marker():
    requirement = _ladybug_requirement()
    assert ";" not in requirement, (
        f"ladybug requirement {requirement!r} carries an environment marker — "
        "markers on this dependency have silently skipped installation on "
        "macOS under packaging >= 25; see the comment in pyproject.toml"
    )


def test_ladybug_dependency_keeps_a_bounded_range():
    requirement = _ladybug_requirement()
    assert re.search(r">=\s*[\d.]+", requirement), "ladybug range lost its floor"
    assert re.search(r"<=?\s*[\d.]+", requirement), "ladybug range lost its ceiling"
