# Dependency license audit

The `dependency-license-audit` job in `release_test.yml` reads **every artifact
the lockfiles can install** and checks what each archive actually contains. It
runs on release validation, meaning pull requests into `main` and manual
dispatch, not on the ordinary pull-request gate.

## Why it reads artifacts instead of metadata

A registry record describes a published package, not its contents. The case that
motivated this: `pypandoc-binary` declares `MIT` on PyPI and in its own wheel
metadata, and ships a 120 MB `pandoc` executable whose terms are GPL. No license
file accompanies it. Any check that reads the declared license alone reports
`MIT` and passes.

A lockfile diff has a second blind spot. A dependency is reviewed once, when it
is added, and never again. `pypandoc-binary` entered the tree through
`unstructured[epub, odt, rtf]` under the `docs` extra and would never be
re-examined.

## What it checks

For each artifact:

1. **The declared license**, read from the artifact's own metadata: a wheel's
   `METADATA`, an sdist's `PKG-INFO`, an npm tarball's `package.json`. It must
   be satisfied by the policy allowlist as an SPDX expression. `OR` accepts an
   allowed option; `AND` requires every part to be allowed. Known free-text
   spellings that predate PEP 639, such as `Apache 2.0` or `BSD 3-Clause
   License`, are translated to their SPDX identifier first, so the output is
   about licenses rather than punctuation. A value is only ever translated to
   the same license it names, never a more permissive one, and a bare `BSD`
   stays ambiguous as `BSD-2-Clause OR BSD-3-Clause`.
2. **Bundled license texts**: every `LICENSE`, `COPYING`, `COPYRIGHT` and
   `NOTICE` file in the archive is scanned for copyleft and source-available
   terms that a permissive declaration would not cover.
3. **Bundled programs**: opaque executables that are not Python extension
   modules or console-script shims. A shebang script or a data file is readable
   source covered by the package's own license, so it is not treated as one. A
   real binary is the payload registry metadata is least likely to describe, so
   its bytes are scanned for the license notices it embeds. This is what
   identifies pandoc as GPL, and it also surfaces NVIDIA's proprietary `ptxas`
   and `cuobjdump` inside `triton`.

Anything unresolved fails the job and is listed as `REVIEW REQUIRED`. Unreadable
archives, unknown metadata and unexpected hosts fail closed. The full findings
are uploaded as the `dependency-license-audit` artifact.

## How it stays affordable

The lock names about 8,700 artifacts totalling more than 50 GB. Wheels are read
**in place** over HTTP range requests: the audit fetches the archive index and
the few small members it needs, so a 25 MB wheel costs a few kilobytes. Only
tarballs, meaning npm packages and the handful of releases that publish no
wheel, are transferred whole. Program bytes are streamed and scanning stops as
soon as every marker is found. Nothing is installed and no packaging or build
hook is ever executed.

## Policy

`.github/dependency-license-policy.json` holds three lists:

- `allowed_licenses`: permissive SPDX identifiers. This is a conservative review
  policy, not a claim that every unlisted license is incompatible with
  Apache-2.0.
- `artifact_hosts`: the hosts artifacts may be downloaded from. Anything else
  requires review.
- `reviewed_packages`: recorded maintainer decisions.

## Reviewing a finding

1. Inspect the exact distribution's license files and bundled components, not
   only the top-level package license. Check attribution, redistribution and
   source-availability obligations for Cognee's intended use and distribution.
2. Prefer a dependency with clear, compatible terms. Dropping the offending
   extra, or switching to a package that does not vendor the component, is
   usually cheaper than carrying the obligation.
3. If the dependency is kept, add an entry to `reviewed_packages` with
   `package` (`pypi:name@version` or `npm:name@version`), a nonempty `reason`,
   and an HTTPS `review_url` recording maintainer approval. Exceptions are
   scoped to that exact version and do not carry to later ones.
4. Policy and exception changes require maintainer review. Never broaden the
   allowlist just to silence a finding.

Run locally from the repository root:

```sh
uv run --script scripts/check_dependency_licenses.py --report license-audit.json
uv run --no-project --python 3.12 --with license-expression==30.4.4 python -m unittest discover -s scripts/tests -p 'test_dependency_licenses.py' -v
```

## Limits

This is an artifact audit, not a legal opinion. It reads the license texts and
notices a distribution carries; it cannot tell you whether an obligation is
discharged, and a passing audit does not waive notice or attribution duties.

Not covered: npm packages bundled inside a parent tarball are read as part of
that parent rather than separately, OS and container packages, model weights,
anything downloaded at runtime, and source distributions for releases that also
publish a wheel, since uv installs the wheel. Release and distribution review
must inspect those separately.
