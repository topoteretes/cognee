# /// script
# requires-python = ">=3.12"
# dependencies = ["license-expression==30.4.4"]
# ///
"""Review added/updated locked dependencies without installing their code."""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import tomllib
from license_expression import ExpressionError, get_spdx_licensing

LICENSING = get_spdx_licensing()


@dataclass(frozen=True, order=True)
class Package:
    ecosystem: str
    name: str
    version: str
    source: str
    declared_license: str = ""

    @property
    def key(self):
        return f"{self.ecosystem}:{self.name}@{self.version}"


def git(*args):
    return subprocess.check_output(["git", *args], text=True)


def packages_from_lock(path, content):
    """Include all locked extras/platforms, not just the CI runner's environment."""
    packages = set()
    if Path(path).name == "uv.lock":
        for item in tomllib.loads(content)["package"]:
            source = item["source"]
            # Only the two first-party workspace roots are exempt from review.
            if source in ({"editable": "."}, {"virtual": "."}) and item["name"] in {
                "cognee",
                "cognee-mcp",
            }:
                continue
            packages.add(
                Package("pypi", item["name"], item["version"], json.dumps(source, sort_keys=True))
            )
    else:
        data = json.loads(content)
        if data.get("lockfileVersion") not in (2, 3):
            raise ValueError(f"Unsupported npm lockfile version: {path}")
        for location, item in data["packages"].items():
            if not location:
                continue
            name = item.get("name") or location.rsplit("node_modules/", 1)[-1]
            packages.add(
                Package(
                    "npm",
                    name,
                    item.get("version", "unknown"),
                    item.get("resolved", ""),
                    item.get("license", ""),
                )
            )
    return packages


def inventory(ref):
    result = {}
    for path in git("ls-tree", "-r", "--name-only", ref).splitlines():
        if Path(path).name in {"uv.lock", "package-lock.json"}:
            result[path] = packages_from_lock(path, git("show", f"{ref}:{path}"))
    return result


def additions(base, head):
    # Compare each lock independently: moving a dependency into MCP is reviewed too.
    return set().union(*(packages - base.get(path, set()) for path, packages in head.items()))


def allowed_expression(expression, allowed):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        return False
    try:
        parsed = LICENSING.parse(expression, validate=True, strict=True)
        values = {
            symbol: LICENSING.TRUE if str(symbol) in allowed else LICENSING.FALSE
            for symbol in parsed.get_symbols()
        }
        return parsed.subs(values).simplify() == LICENSING.TRUE
    except (ExpressionError, ValueError):
        return False


def declared_license(package):
    """Read exact-version registry metadata; do not execute package build hooks."""
    if package.ecosystem == "npm":
        if not package.source.startswith("https://registry.npmjs.org/"):
            raise ValueError("non-registry dependency requires source review")
        return package.declared_license
    if json.loads(package.source) != {"registry": "https://pypi.org/simple"}:
        raise ValueError("non-PyPI dependency requires source review")
    url = f"https://pypi.org/pypi/{quote(package.name, safe='')}/{quote(package.version, safe='')}/json"
    with urlopen(url, timeout=30) as response:
        info = json.load(response)["info"]
    expression = info.get("license_expression") or info.get("license")
    if expression:
        return expression
    # Legacy classifiers only identify a few SPDX licenses unambiguously.
    classifiers = [value for value in info.get("classifiers", []) if value.startswith("License ::")]
    aliases = {
        "License :: OSI Approved :: MIT License": "MIT",
        "License :: OSI Approved :: Apache Software License": "Apache-2.0",
        "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    }
    return aliases.get(classifiers[0], "") if len(classifiers) == 1 else ""


def review(packages, policy, lookup=declared_license):
    failures = []
    for package in sorted(packages):
        # Exceptions are exact-version AND source scoped; record a review URL/reason.
        approved = next(
            (
                entry
                for entry in policy["reviewed_packages"]
                if entry["package"] == package.key
                and entry["source"] == package.source
                and entry["reason"].strip()
                and entry["review_url"].startswith("https://")
            ),
            None,
        )
        if approved:
            print(f"REVIEWED {package.key}: {approved['review_url']}")
            continue
        try:
            expression = lookup(package)
            if not allowed_expression(expression, set(policy["allowed_licenses"])):
                raise ValueError(f"license needs review: {str(expression or 'UNKNOWN')[:160]!r}")
            print(f"PASS {package.key}: {expression}")
        except (OSError, ValueError) as error:
            # Unknown metadata, network errors and unsupported sources fail closed.
            failures.append(f"{package.key}: {error}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", default="HEAD", help="Head commit SHA")
    args = parser.parse_args()
    policy = json.loads(Path(".github/dependency-license-policy.json").read_text())
    changed = additions(inventory(args.base), inventory(args.head))
    print(f"Reviewing {len(changed)} added/updated locked dependencies.")
    failures = review(changed, policy)
    for failure in failures:
        print(f"REVIEW REQUIRED: {failure}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
