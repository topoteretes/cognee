"""Offline regression tests for the dependency license gate."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_dependency_licenses import (
    Package,
    additions,
    allowed_expression,
    declared_license,
    packages_from_lock,
    review,
)

POLICY = {"allowed_licenses": ["MIT", "Apache-2.0"], "reviewed_packages": []}
SOURCE = '{"registry": "https://pypi.org/simple"}'
PACKAGE = Package("pypi", "example", "1.0", SOURCE)


class LicenseTests(unittest.TestCase):
    def test_permissive_expression(self):
        for expression in ["MIT", "MIT OR GPL-3.0-only", "(MIT AND Apache-2.0) OR GPL-3.0-only"]:
            with self.subTest(expression=expression):
                self.assertTrue(allowed_expression(expression, POLICY["allowed_licenses"]))

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
                self.assertFalse(allowed_expression(expression, POLICY["allowed_licenses"]))

    def test_unknown_or_failed_lookup_blocks(self):
        self.assertEqual(len(review({PACKAGE}, POLICY, lambda package: "")), 1)

        def unavailable(package):
            raise OSError("Registry unavailable")

        self.assertEqual(len(review({PACKAGE}, POLICY, unavailable)), 1)

    def test_approved_package_is_version_and_source_specific(self):
        policy = {
            **POLICY,
            "reviewed_packages": [
                {
                    "package": PACKAGE.key,
                    "source": SOURCE,
                    "reason": "Reviewed distribution terms",
                    "review_url": "https://example.com/review",
                }
            ],
        }
        self.assertEqual(review({PACKAGE}, policy, lambda package: "UNKNOWN"), [])
        for other in [
            Package("pypi", "example", "2.0", SOURCE),
            Package("pypi", "example", "1.0", "other-source"),
        ]:
            self.assertEqual(len(review({other}, policy, lambda package: "UNKNOWN")), 1)

    def test_empty_exception_reason_does_not_bypass(self):
        policy = {
            **POLICY,
            "reviewed_packages": [
                {
                    "package": PACKAGE.key,
                    "source": SOURCE,
                    "reason": "",
                    "review_url": "https://example.com/review",
                }
            ],
        }
        self.assertEqual(len(review({PACKAGE}, policy, lambda package: "UNKNOWN")), 1)

    def test_non_pypi_source_cannot_borrow_pypi_metadata(self):
        package = Package("pypi", "example", "1.0", '{"git": "https://example.com/repo"}')
        with self.assertRaisesRegex(ValueError, "source review"):
            declared_license(package)

    def test_npm_source_and_license(self):
        package = Package(
            "npm", "example", "1.0", "https://registry.npmjs.org/example/1.tgz", "MIT"
        )
        self.assertEqual(declared_license(package), "MIT")
        self.assertFalse(review({package}, POLICY))
        with self.assertRaises(ValueError):
            declared_license(Package("npm", "example", "1.0", "file:../example", "MIT"))

    def test_pypi_expression_takes_precedence(self):
        with patch("check_dependency_licenses.urlopen") as request:
            response = request.return_value.__enter__.return_value
            response.read.return_value = json.dumps(
                {"info": {"license_expression": "GPL-3.0-only", "license": "MIT"}}
            )
            self.assertEqual(declared_license(PACKAGE), "GPL-3.0-only")
            self.assertIn("/example/1.0/json", request.call_args.args[0])

    def test_uv_all_versions_and_platforms_are_included(self):
        lock = """version = 1
[[package]]
name = "cognee"
version = "1.6.0"
source = { editable = "." }
[[package]]
name = "example"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "example"
version = "2.0"
source = { registry = "https://pypi.org/simple" }
"""
        packages = packages_from_lock("uv.lock", lock)
        self.assertEqual({package.version for package in packages}, {"1.0", "2.0"})

    def test_npm_scoped_transitive_and_dev_packages(self):
        lock = {
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "frontend"},
                "node_modules/@org/name": {"version": "1.0", "license": "MIT", "dev": True},
                "node_modules/parent/node_modules/child": {"version": "2.0"},
            },
        }
        packages = packages_from_lock("package-lock.json", json.dumps(lock))
        self.assertEqual({package.name for package in packages}, {"@org/name", "child"})

    def test_diff_reviews_version_changes_and_new_lockfiles(self):
        updated = Package("pypi", "example", "2.0", SOURCE)
        base = {"uv.lock": {PACKAGE}}
        self.assertEqual(additions(base, base), set())
        self.assertEqual(additions(base, {"uv.lock": {updated}}), {updated})
        self.assertEqual(additions(base, {**base, "cognee-mcp/uv.lock": {PACKAGE}}), {PACKAGE})
        self.assertEqual(additions(base, {}), set())


if __name__ == "__main__":
    unittest.main()
