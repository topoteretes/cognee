"""Offline regression tests for the dependency license audit."""

import io
import json
import sys
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_dependency_licenses as audit_module
from check_dependency_licenses import (
    Artifact,
    Inspection,
    allowed_expression,
    approval,
    artifacts_from_lock,
    audit,
    inspect,
    is_program,
    metadata_license,
    normalize,
    npm_license,
    problems,
    restricted_in,
    scan_stream,
)

ALLOWED = {"MIT", "Apache-2.0"}
POLICY = {
    "allowed_licenses": sorted(ALLOWED),
    "artifact_hosts": ["files.pythonhosted.org", "registry.npmjs.org"],
    "reviewed_packages": [],
}
WHEEL = Artifact(
    "pypi", "example", "1.0", "wheel", "https://files.pythonhosted.org/x/example-1.0.whl", 10
)


class ExpressionTests(unittest.TestCase):
    def test_permissive_expression(self):
        for expression in ["MIT", "MIT OR GPL-3.0-only", "(MIT AND Apache-2.0) OR GPL-3.0-only"]:
            with self.subTest(expression=expression):
                self.assertTrue(allowed_expression(expression, ALLOWED))

    def test_review_required(self):
        for expression in [
            "GPL-3.0-only",
            "AGPL-3.0-only",
            "LGPL-2.1-only",
            "MPL-2.0",
            "MIT AND GPL-3.0-only",
            "GPL-2.0-only WITH Classpath-exception-2.0",
            "UNKNOWN",
            "LicenseRef-Custom",
            "MIT OR",
            "",
            None,
        ]:
            with self.subTest(expression=expression):
                self.assertFalse(allowed_expression(expression, ALLOWED))


class NormalizeTests(unittest.TestCase):
    """Legacy free-text values are spelling, not a license question."""

    def test_known_free_text_maps_to_spdx(self):
        for text, expected in [
            ("MIT License", "MIT"),
            ("Apache 2.0", "Apache-2.0"),
            ("Apache License, Version 2.0", "Apache-2.0"),
            ("  apache software license  ", "Apache-2.0"),
            ("BSD 3-Clause License", "BSD-3-Clause"),
            ("ISC", "ISC"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)
                self.assertTrue(
                    allowed_expression(text, {"MIT", "Apache-2.0", "BSD-3-Clause", "ISC"})
                )

    def test_bare_bsd_is_ambiguous_and_stays_ambiguous(self):
        """ "BSD" names either variant, so accept it only if both are allowed."""
        self.assertEqual(normalize("BSD"), "BSD-2-Clause OR BSD-3-Clause")
        self.assertTrue(allowed_expression("BSD", {"BSD-2-Clause", "BSD-3-Clause"}))
        self.assertTrue(allowed_expression("BSD", {"BSD-3-Clause"}))
        self.assertFalse(allowed_expression("BSD", {"MIT"}))

    def test_undeclared_placeholders_are_not_a_license(self):
        for text in ["UNKNOWN", "unknown", "None", "Other/Proprietary License"]:
            with self.subTest(text=text):
                self.assertEqual(normalize(text), "")
                self.assertFalse(allowed_expression(text, {"MIT"}))

    def test_restrictive_free_text_is_never_widened(self):
        for text in ["MPL 2.0", "LGPL-3.0-or-later", "LicenseRef-NVIDIA-Proprietary"]:
            with self.subTest(text=text):
                self.assertFalse(allowed_expression(text, {"MIT", "Apache-2.0", "BSD-3-Clause"}))

    def test_unrecognised_text_is_passed_through_unchanged(self):
        self.assertEqual(normalize("某 custom license"), "某 custom license")


class MetadataTests(unittest.TestCase):
    def test_expression_beats_legacy_and_classifier(self):
        text = "Name: x\nLicense-Expression: Apache-2.0\nLicense: MIT\n"
        self.assertEqual(metadata_license(text), "Apache-2.0")

    def test_single_classifier_is_translated(self):
        text = "Name: x\nClassifier: License :: OSI Approved :: MIT License\n"
        self.assertEqual(metadata_license(text), "MIT")

    def test_ambiguous_classifiers_are_not_guessed(self):
        text = (
            "Name: x\n"
            "Classifier: License :: OSI Approved :: MIT License\n"
            "Classifier: License :: OSI Approved :: Apache Software License\n"
        )
        self.assertEqual(metadata_license(text), "")

    def test_body_after_the_headers_is_ignored(self):
        text = "Name: x\nLicense: MIT\n\nREADME\nLicense: GPL-3.0-only\n"
        self.assertEqual(metadata_license(text), "MIT")

    def test_npm_manifest_forms(self):
        self.assertEqual(npm_license(json.dumps({"license": "MIT"})), "MIT")
        self.assertEqual(npm_license(json.dumps({"license": {"type": "MIT"}})), "MIT")
        legacy = json.dumps({"licenses": [{"type": "MIT"}, {"type": "Apache-2.0"}]})
        self.assertEqual(npm_license(legacy), "MIT OR Apache-2.0")
        self.assertEqual(npm_license(json.dumps({})), "")


class ContentTests(unittest.TestCase):
    def test_restricted_text_is_detected_through_wrapping(self):
        text = "        GNU GENERAL\n   PUBLIC     LICENSE\nVersion 2, June 1991"
        self.assertEqual(restricted_in(text), {"GPL-3.0-only"})
        self.assertEqual(restricted_in("Permission is hereby granted, free of charge"), set())

    def test_program_detection_targets_vendored_executables(self):
        self.assertTrue(is_program("pypandoc/files/pandoc", 0o755))
        self.assertTrue(is_program("pkg/bin/tool.exe", 0o755))
        for name, mode in [
            ("pkg/_speedups.so", 0o755),  # a Python extension module
            ("pkg/_speedups.pyd", 0o755),
            ("pkg/module.py", 0o755),
            ("example-1.0.data/scripts/cli", 0o755),  # a console-script shim
            ("example-1.0.dist-info/RECORD", 0o755),
            ("LICENSE", 0o755),  # a license file is never a program
            ("vendor/NOTICE", 0o755),
            ("pypandoc/files/pandoc", 0o644),  # not executable
        ]:
            with self.subTest(name=name, mode=oct(mode)):
                self.assertFalse(is_program(name, mode))

    def test_scan_finds_a_marker_split_across_chunks(self):
        payload = b"\x00" * 30 + b"GNU GENERAL PUBLIC LICENSE" + b"\x00" * 30
        with patch.object(audit_module, "SCAN_CHUNK", 40):
            self.assertEqual(scan_stream(io.BytesIO(payload)), {"GPL-3.0-only"})

    def test_scan_of_a_clean_binary_finds_nothing(self):
        self.assertEqual(scan_stream(io.BytesIO(b"\x7fELF" + b"\x00" * 4096)), set())

    def test_text_payloads_are_not_vendored_programs(self):
        """A readable script's license is its package's, so it is not a finding."""
        for payload in [
            b"#!/usr/bin/env node\nrequire('../lib/cli')\n",  # an npm bin shim
            b"#!/bin/sh\nnpx lint-staged\n",  # a git hook
            b'{"freq": {"a": 1}}\n' * 100,  # a data file with an exec bit
        ]:
            with self.subTest(payload=payload[:20]):
                self.assertIsNone(scan_stream(io.BytesIO(payload)))


class InventoryTests(unittest.TestCase):
    UV_LOCK = """version = 1
[[package]]
name = "cognee"
version = "1.6.0"
source = { editable = "." }
[[package]]
name = "wheeled"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/a/wheeled-1.0.tar.gz", size = 5 }
wheels = [
    { url = "https://files.pythonhosted.org/a/wheeled-1.0-linux.whl", size = 7 },
    { url = "https://files.pythonhosted.org/a/wheeled-1.0-macos.whl", size = 8 },
]
[[package]]
name = "sourceonly"
version = "2.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/b/sourceonly-2.0.tar.gz", size = 9 }
"""

    def test_every_platform_wheel_is_audited(self):
        found = artifacts_from_lock("uv.lock", self.UV_LOCK)
        self.assertEqual(
            {(a.name, a.kind) for a in found},
            {("wheeled", "wheel"), ("wheeled", "wheel"), ("sourceonly", "sdist")},
        )
        self.assertEqual(len([a for a in found if a.name == "wheeled"]), 2)

    def test_first_party_roots_are_skipped(self):
        self.assertNotIn("cognee", {a.name for a in artifacts_from_lock("uv.lock", self.UV_LOCK)})

    def test_non_registry_python_source_needs_review(self):
        lock = """version = 1
[[package]]
name = "vendored"
version = "1.0"
source = { git = "https://example.com/repo" }
"""
        with self.assertRaisesRegex(ValueError, "requires manual review"):
            artifacts_from_lock("uv.lock", lock)

    def test_npm_transitive_and_dev_packages_are_audited(self):
        lock = {
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "frontend"},
                "node_modules/@org/name": {
                    "version": "1.0",
                    "dev": True,
                    "resolved": "https://registry.npmjs.org/@org/name/-/name-1.0.tgz",
                },
                "node_modules/parent/node_modules/child": {
                    "version": "2.0",
                    "resolved": "https://registry.npmjs.org/child/-/child-2.0.tgz",
                },
                # Ships inside its parent's tarball, which is audited whole.
                "node_modules/parent/node_modules/bundled": {"version": "3.0", "inBundle": True},
                "node_modules/linked": {"link": True},
            },
        }
        found = artifacts_from_lock("package-lock.json", json.dumps(lock))
        self.assertEqual({a.name for a in found}, {"@org/name", "child"})

    def test_non_registry_npm_source_needs_review(self):
        lock = {
            "lockfileVersion": 3,
            "packages": {"node_modules/x": {"version": "1.0", "resolved": "file:../x"}},
        }
        with self.assertRaisesRegex(ValueError, "requires manual review"):
            artifacts_from_lock("package-lock.json", json.dumps(lock))

    def test_unsupported_npm_lockfile_version(self):
        with self.assertRaisesRegex(ValueError, "unsupported npm lockfile version"):
            artifacts_from_lock("package-lock.json", json.dumps({"lockfileVersion": 1}))


def build_tarball(root: str, files: dict) -> bytes:
    """Pack an archive shaped like a real sdist or npm tarball."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, (body, mode) in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size, info.mode = len(body), mode
            archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


class TarballTests(unittest.TestCase):
    """The reader must find a manifest nested under the archive's root directory."""

    def read(self, artifact, payload):
        with patch.object(audit_module, "download", return_value=io.BytesIO(payload)):
            return inspect(artifact, set(POLICY["artifact_hosts"]))

    def test_sdist_manifest_license_and_payload(self):
        payload = build_tarball(
            "example-1.0",
            {
                "PKG-INFO": (b"Name: example\nLicense: MIT\n", 0o644),
                "LICENSE": (b"GNU GENERAL PUBLIC LICENSE Version 2", 0o644),
                "bin/tool": (b"\x7fELF\x00\x00 GNU GENERAL PUBLIC LICENSE", 0o755),
            },
        )
        artifact = Artifact(
            "pypi", "example", "1.0", "sdist", "https://files.pythonhosted.org/e/example-1.0.tar.gz"
        )
        found = self.read(artifact, payload)
        self.assertEqual(found.declared, "MIT")
        self.assertEqual(found.bundled, {"LICENSE": ["GPL-3.0-only"]})
        self.assertEqual(found.programs, {"bin/tool": ["GPL-3.0-only"]})

    def test_npm_manifest_is_read_from_the_package_directory(self):
        payload = build_tarball(
            "package", {"package.json": (b'{"name":"x","license":"MIT"}', 0o644)}
        )
        artifact = Artifact("npm", "x", "1.0", "npm", "https://registry.npmjs.org/x/-/x-1.0.tgz")
        found = self.read(artifact, payload)
        self.assertEqual(found.declared, "MIT")
        self.assertEqual(problems(found, ALLOWED), [])


class HostTests(unittest.TestCase):
    def test_artifact_host_must_be_named_by_the_policy(self):
        rogue = Artifact("pypi", "x", "1.0", "wheel", "https://evil.example/x.whl", 10)
        with self.assertRaisesRegex(ValueError, "host requires review"):
            inspect(rogue, set(POLICY["artifact_hosts"]))

    def test_plain_http_is_refused(self):
        insecure = Artifact("pypi", "x", "1.0", "wheel", "http://files.pythonhosted.org/x.whl", 10)
        with self.assertRaisesRegex(ValueError, "host requires review"):
            inspect(insecure, set(POLICY["artifact_hosts"]))


class ProblemTests(unittest.TestCase):
    def test_permissive_artifact_has_no_problems(self):
        self.assertEqual(problems(Inspection(declared="MIT"), ALLOWED), [])

    def test_declared_license_outside_the_allowlist(self):
        issues = problems(Inspection(declared="GPL-3.0-only"), ALLOWED)
        self.assertIn("declared license needs review", issues[0])

    def test_bundled_license_text_is_reported_even_when_the_declaration_passes(self):
        found = Inspection(declared="MIT", bundled={"vendor/COPYING": ["GPL-3.0-only"]})
        issues = problems(found, ALLOWED)
        self.assertEqual(issues, ["bundled license text in vendor/COPYING: GPL-3.0-only"])

    def test_a_program_embedding_restricted_terms_is_reported_with_evidence(self):
        """The pandoc case: MIT metadata, no license file, a GPL binary inside."""
        found = Inspection(declared="MIT", programs={"pypandoc/files/pandoc": ["GPL-3.0-only"]})
        issues = problems(found, ALLOWED)
        self.assertEqual(
            issues,
            ["bundles 1 program(s) embedding GPL-3.0-only: pypandoc/files/pandoc"],
        )

    def test_a_program_with_no_license_text_still_needs_review(self):
        found = Inspection(declared="MIT", programs={"pkg/bin/tool": []})
        self.assertEqual(
            problems(found, ALLOWED),
            ["bundles 1 program(s) with no license text: pkg/bin/tool"],
        )

    def test_many_programs_collapse_into_one_finding_per_package(self):
        """A package shipping its own test binaries must not bury other packages."""
        found = Inspection(declared="MIT", programs={f"torch/test/t{n}": [] for n in range(120)})
        issues = problems(found, ALLOWED)
        self.assertEqual(len(issues), 1)
        self.assertIn("bundles 120 program(s) with no license text", issues[0])
        self.assertIn("and 114 more", issues[0])


class AuditTests(unittest.TestCase):
    def test_findings_are_grouped_by_package_version(self):
        other = Artifact(
            "pypi",
            "example",
            "1.0",
            "wheel",
            "https://files.pythonhosted.org/x/example-1.0-mac.whl",
            10,
        )
        findings = audit(
            {WHEEL, other}, POLICY, jobs=2, reader=lambda a: Inspection(declared="GPL-3.0-only")
        )
        self.assertEqual(list(findings), ["pypi:example@1.0"])
        issue = next(iter(findings["pypi:example@1.0"]))
        self.assertEqual(
            sorted(findings["pypi:example@1.0"][issue]), sorted([WHEEL.filename, other.filename])
        )

    def test_a_clean_artifact_produces_no_finding(self):
        self.assertEqual(audit({WHEEL}, POLICY, jobs=1, reader=lambda a: Inspection("MIT")), {})

    def test_an_unreadable_artifact_fails_closed(self):
        def unreadable(artifact):
            raise OSError("registry unavailable")

        findings = audit({WHEEL}, POLICY, jobs=1, reader=unreadable)
        self.assertIn("could not be read", next(iter(findings["pypi:example@1.0"])))


class ApprovalTests(unittest.TestCase):
    ENTRY = {
        "package": "pypi:example@1.0",
        "reason": "Reviewed distribution terms",
        "review_url": "https://example.com/review",
    }

    def policy(self, **overrides):
        return {**POLICY, "reviewed_packages": [{**self.ENTRY, **overrides}]}

    def test_exception_applies_to_the_reviewed_version_only(self):
        self.assertIsNotNone(approval("pypi:example@1.0", self.policy()))
        self.assertIsNone(approval("pypi:example@2.0", self.policy()))

    def test_an_exception_without_a_reason_does_not_bypass(self):
        self.assertIsNone(approval("pypi:example@1.0", self.policy(reason="  ")))

    def test_an_exception_without_a_review_url_does_not_bypass(self):
        self.assertIsNone(approval("pypi:example@1.0", self.policy(review_url="internal note")))


class PolicyFileTests(unittest.TestCase):
    def test_the_shipped_policy_is_well_formed(self):
        path = Path(__file__).resolve().parents[2] / ".github/dependency-license-policy.json"
        policy = json.loads(path.read_text())
        self.assertTrue(set(policy) >= {"allowed_licenses", "artifact_hosts", "reviewed_packages"})
        for expression in policy["allowed_licenses"]:
            with self.subTest(expression=expression):
                self.assertTrue(allowed_expression(expression, set(policy["allowed_licenses"])))
        for entry in policy["reviewed_packages"]:
            with self.subTest(package=entry.get("package")):
                self.assertIsNotNone(approval(entry["package"], policy))


if __name__ == "__main__":
    unittest.main()
