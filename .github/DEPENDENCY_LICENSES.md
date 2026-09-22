# Dependency license review

The `dependency-license-check` job reviews additions and version/source changes
in every tracked `uv.lock` and `package-lock.json`. Python coverage includes
transitive dependencies, all locked extras and platform variants; npm coverage
includes development dependencies. Python manifests must match their lockfiles.
The license scanner does not install or execute dependency code.

The policy allows common permissive SPDX licenses. Other licenses, unknown or
malformed metadata, unsupported sources and registry lookup failures fail the
check and require review. This is a conservative review policy, not a claim that
every unlisted license is incompatible with Apache-2.0. SPDX `OR` accepts an
allowed licensing option; `AND` requires all obligations to be allowed.

## Reviewing a dependency

1. Keep manifests and lockfiles in sync when changing dependencies.
2. Inspect the exact distribution's license files and bundled components, not
   only the top-level package license. Check attribution, redistribution and
   source-availability obligations for Cognee's intended use and distribution.
3. Prefer a dependency with clear, compatible terms. If approval needs an
   exception, add an entry to `.github/dependency-license-policy.json` with
   `package` (`pypi:name@version` or `npm:name@version`), `source` (the exact
   lockfile source serialized as sorted JSON for Python, or npm's resolved URL),
   a nonempty `reason`, and an HTTPS `review_url` recording maintainer approval.
   Exceptions do not automatically apply to future versions or changed sources.
4. Policy and exception changes require maintainer review; never broaden the
   allowlist just to silence a failing dependency.

Run locally from the repository root:

```sh
uv run --script scripts/check_dependency_licenses.py --base origin/dev --head HEAD
uv run --no-project --python 3.12 --with license-expression==30.4.4 python -m unittest discover -s scripts/tests -p 'test_dependency_licenses.py' -v
```

## Enforcement and limits

Make `dependency-license-check` a required status check on protected branches
after enabling this workflow. The workflow itself does not change branch rules.
Fork PRs run with a read-only token and no repository secrets.

This is an incremental metadata gate, not a legal opinion or a full artifact
audit. PyPI's exact-release metadata and npm lockfile declarations can omit or
misdescribe bundled binaries, vendored code or notices. Unchanged locked packages,
new activation of an already locked extra, npm manifest/lockfile drift, OS/container
packages, model weights and runtime downloads are not audited by this check.
Release/distribution review must inspect those separately. A passing check does
not waive notice, attribution or other compliance obligations.
