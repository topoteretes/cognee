"""The license check fails on strong copyleft cognee would inherit, and only on that."""

import importlib.util
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "check_license_compatibility", ROOT / "scripts/check_license_compatibility.py"
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

ELF = b"\x7fELF" + b"\0" * 60


def make_dist(root, name, metadata="", files=None):
    """Install a fake distribution under ``root``: dist-info, RECORD and ``files``."""
    dist_info = root / f"{name}-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: {name}\nVersion: 1.0\n{metadata}"
    )
    records = [f"{dist_info.name}/METADATA,,"]
    for path, content in (files or {}).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append(f"{path},,")
    (dist_info / "RECORD").write_text("\n".join(records) + "\n")
    return PathDistribution(dist_info)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("GPL-3.0-only", {"GPL"}),
        ("GPL-2.0-or-later", {"GPL"}),
        ("GPLv2+", {"GPL"}),
        ("AGPL-3.0-or-later", {"AGPL"}),
        ("LGPL-2.1-only", set()),
        ("LGPL with exceptions", set()),
        ("MIT OR GPL-2.0-only", set()),
        ("FTL OR GPL-2.0-or-later", set()),
        ("GPL-3.0-or-later WITH GCC-exception-3.1", set()),
        ("GPL-compatible", set()),
        ("MPL-2.0", set()),
        ("Apache-2.0", set()),
        ("LicenseRef-NVIDIA-Proprietary", set()),
    ],
)
def test_expressions(expression, expected):
    assert check.strong_copyleft([expression]) == expected


@pytest.mark.parametrize(
    "classifiers, expected",
    [
        (["OSI Approved :: GNU General Public License v3 (GPLv3)"], {"GPL"}),
        (["OSI Approved :: GNU Affero General Public License v3"], {"AGPL"}),
        (["OSI Approved :: GNU Library or Lesser General Public License (LGPL)"], set()),
        (["OSI Approved :: GNU General Public License v2 or later (GPLv2+)"], {"GPL"}),
        (["OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"], set()),
        # Several license classifiers are alternatives: dual licensing.
        (
            ["OSI Approved :: MIT License", "OSI Approved :: GNU General Public License (GPL)"],
            set(),
        ),
    ],
)
def test_classifiers(classifiers, expected):
    assert check.strong_copyleft(classifiers) == expected


@pytest.mark.parametrize(
    "metadata, files",
    [
        pytest.param("License-Expression: GPL-3.0-only\n", {}, id="expression"),
        pytest.param("License: AGPLv3\n", {}, id="license-field"),
        pytest.param(
            "Classifier: License :: OSI Approved :: GNU General Public License v2 (GPLv2)\n",
            {},
            id="classifier",
        ),
        pytest.param(
            # A permissive License field must not excuse a GPL classifier.
            "License: MIT\n"
            "Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)\n",
            {},
            id="classifier-despite-permissive-field",
        ),
        pytest.param(
            "License: BSD-3-Clause\n",
            {
                "pkg-1.0.dist-info/licenses/LICENSES_bundled.txt": b"Name: libx\nLicense: GPL-2.0-only\n"
            },
            id="bundled-manifest",
        ),
        pytest.param(
            "License: MIT\n",
            {"pkg.libs/libx-1a2b.so.1": ELF + b"GNU General Public License"},
            id="vendored-library",
        ),
        pytest.param(
            "License: MIT\n",
            {"pkg/.dylibs/libx.dylib": ELF + b"GNU AFFERO GENERAL PUBLIC LICENSE"},
            id="vendored-dylib",
        ),
    ],
)
def test_flags_strong_copyleft(tmp_path, metadata, files):
    assert check.findings(make_dist(tmp_path, "pkg", metadata, files))


@pytest.mark.parametrize(
    "metadata, files",
    [
        pytest.param("License-Expression: Apache-2.0\n", {}, id="permissive"),
        pytest.param("License-Expression: LGPL-2.1-only\n", {}, id="lgpl"),
        pytest.param("License-Expression: MPL-2.0\n", {}, id="mpl"),
        pytest.param("License-Expression: LicenseRef-NVIDIA-Proprietary\n", {}, id="proprietary"),
        pytest.param(
            # pypandoc-binary: a GPL program run as a subprocess is aggregation.
            "License: MIT\n",
            {"pkg/files/pandoc": ELF + b"GNU GENERAL PUBLIC LICENSE"},
            id="bundled-program",
        ),
        pytest.param(
            # LGPL texts quote the GPL's title.
            "License: MIT\n",
            {
                "pkg.libs/libx.so.1": ELF
                + b"GNU LESSER GENERAL PUBLIC LICENSE ... the GNU General Public License"
            },
            id="vendored-lgpl-library",
        ),
        pytest.param(
            # The package's own extension module is not a vendored library.
            "License: MIT\n",
            {"pkg/_ext.so": ELF + b"GNU GENERAL PUBLIC LICENSE"},
            id="own-extension",
        ),
        pytest.param(
            "License: BSD-3-Clause\n",
            {
                "pkg-1.0.dist-info/licenses/LICENSE.txt": b"Name: libgfortran\n"
                b"License: GPL-3.0-or-later WITH GCC-exception-3.1\n"
            },
            id="gcc-runtime-exception",
        ),
        pytest.param(
            "License-Expression: MPL-2.0\n",
            {
                "pkg-1.0.dist-info/licenses/LICENSE": b'"Secondary License" means either '
                b"the GNU General Public License, Version 2.0"
            },
            id="license-text-mentioning-gpl",
        ),
    ],
)
def test_ignores_terms_that_leave_cognee_unchanged(tmp_path, metadata, files):
    assert check.findings(make_dist(tmp_path, "pkg", metadata, files)) == []


def test_embedded_notice_across_read_chunks(tmp_path, monkeypatch):
    # The 26-byte notice starts at byte 71 and crosses the 96-byte chunk boundary.
    monkeypatch.setattr(check, "CHUNK", 32)
    dist = make_dist(
        tmp_path,
        "pkg",
        "License: MIT\n",
        {"pkg.libs/libx.so.1": ELF + b"x" * 7 + b"GNU GENERAL PUBLIC LICENSE"},
    )
    assert check.findings(dist) == [("pkg.libs/libx.so.1", "embeds GPL")]


def test_main_fails_on_findings_and_honours_reviewed(tmp_path, monkeypatch, capsys):
    gpl = make_dist(tmp_path, "gpl-pkg", "License-Expression: GPL-3.0-only\n")
    clean = make_dist(tmp_path, "clean-pkg", "License-Expression: MIT\n")
    monkeypatch.setattr(check, "distributions", lambda: [gpl, clean])

    assert check.main() == 1
    assert "COPYLEFT gpl-pkg: License-Expression declares GPL" in capsys.readouterr().err

    monkeypatch.setattr(check, "REVIEWED", {"gpl-pkg": "test"})
    assert check.main() == 0

    monkeypatch.setattr(check, "distributions", lambda: [clean])
    monkeypatch.setattr(check, "REVIEWED", {})
    assert check.main() == 0
