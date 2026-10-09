"""Guards the ladybug macOS version-split markers against evaluator drift.

The two ladybug dependency lines partition environments: macOS 13/14
(Darwin 22/23) get 0.17.x (the last line with macosx_13_0 wheels), everything
else gets the newest supported range. The markers may use ONLY substring
operators ('in'/'not in' on platform_version): those are plain string
operations in every marker evaluator, while ordering comparisons on
platform_release are unexpressible across generations — packaging >= 25
(vendored in current pip) silently evaluated the old ``platform_release >=
'24.'`` marker False on macOS and skipped ladybug entirely, and valid version
literals crash packaging <= 24 on Linux kernel strings.

These tests evaluate the real markers from pyproject.toml against canonical
environments with every packaging implementation importable here (the
standalone library and pip's vendored copy), so an evaluator upgrade that
changes the partition fails CI instead of shipping.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

ENVIRONMENTS = {
    "linux": {
        "sys_platform": "linux",
        "platform_version": "#40~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov 14 12:00:00 UTC 2",
        "platform_release": "6.8.0-45-generic",
    },
    "windows": {
        "sys_platform": "win32",
        "platform_version": "10.0.22621",
        "platform_release": "10",
    },
    "mac13": {
        "sys_platform": "darwin",
        "platform_version": "Darwin Kernel Version 22.6.0: Wed Jul  5 22:22:05 PDT 2023; root:xnu-8796",
        "platform_release": "22.6.0",
    },
    "mac14": {
        "sys_platform": "darwin",
        "platform_version": "Darwin Kernel Version 23.6.0: Mon Jul 29 21:14:30 PDT 2024; root:xnu-10063",
        "platform_release": "23.6.0",
    },
    "mac15": {
        "sys_platform": "darwin",
        "platform_version": "Darwin Kernel Version 24.0.0: Mon Aug 12 20:51:54 PDT 2024; root:xnu-11215",
        "platform_release": "24.0.0",
    },
    "mac_future": {
        "sys_platform": "darwin",
        "platform_version": "Darwin Kernel Version 28.1.0: Tue Jan  5 10:00:00 PST 2027; root:xnu-99999",
        "platform_release": "28.1.0",
    },
}

# Which dependency line must be active where. Old macs get the 0.17 line;
# everything else (including future macOS versions — the enumerated set is
# closed) gets the newest line. Never both: the ranges conflict.
OLD_MAC_ENVIRONMENTS = {"mac13", "mac14"}


def _ladybug_requirements() -> list[str]:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    requirements = re.findall(r'"(ladybug[^"]*)"', text)
    assert len(requirements) == 2, f"expected the two split ladybug lines, found {requirements}"
    return requirements


def _marker_backends():
    import packaging.markers as standalone

    backends = [("packaging", standalone)]
    try:
        import pip._vendor.packaging.markers as vendored

        backends.append(("pip-vendored packaging", vendored))
    except ImportError:
        pass
    return backends


def _split_lines(requirements):
    old_mac_line = next(r for r in requirements if "<0.18" in r.replace(" ", ""))
    newest_line = next(r for r in requirements if r is not old_mac_line)
    return old_mac_line, newest_line


def test_markers_use_only_substring_operators():
    for requirement in _ladybug_requirements():
        marker = requirement.split(";", 1)[1]
        assert not re.search(r"platform_release", marker), (
            f"ordering on platform_release is unexpressible across evaluator "
            f"generations; use substring markers on platform_version: {requirement!r}"
        )


@pytest.mark.parametrize("environment_name", sorted(ENVIRONMENTS))
def test_partition_across_marker_evaluators(environment_name):
    environment = ENVIRONMENTS[environment_name]
    old_mac_line, newest_line = _split_lines(_ladybug_requirements())
    expect_old = environment_name in OLD_MAC_ENVIRONMENTS

    for backend_name, markers_module in _marker_backends():
        old_active = markers_module.Marker(old_mac_line.split(";", 1)[1]).evaluate(environment)
        newest_active = markers_module.Marker(newest_line.split(";", 1)[1]).evaluate(environment)
        assert old_active == expect_old, (
            f"[{backend_name}] 0.17 line wrong on {environment_name}: {old_active}"
        )
        assert newest_active == (not expect_old), (
            f"[{backend_name}] newest line wrong on {environment_name}: {newest_active}"
        )
