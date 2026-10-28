# How to make a release

Every package under `cognee-connectors/` is released on its own:
`cognee-connectors-core` and each `cognee-connector-<provider>` have their own
version in their own `pyproject.toml`, their own git tag `<distribution>-v<version>`
and their own PyPI project. They version independently of the cognee library, as
cognee-mcp does. The workspace root is never published.

Releases run from
[`release_connectors.yml`](https://github.com/topoteretes/cognee/actions/workflows/release_connectors.yml)
in the cognee repository. One run releases either one package or, with `all`,
every package whose current version is not on PyPI yet.

## How releases are triggered

- **With cognee.** Every stable cognee release (`release.yml` on `main`, not a
  `.devN` canary or an `rc`) ends with the job `release-connectors`, which
  dispatches `release_connectors.yml` with `connector=all` on the same branch.
  Packages whose version did not change are skipped, so the run publishes exactly
  the connector bumps merged since the last one. Turn it off for one cognee
  release with the `release_connectors` input. A connector failure shows up in
  the connector run and never fails the cognee release.
- **On their own.** Run `release_connectors.yml` by hand with one package, or with
  `all`, whenever a connector fix should not wait for the next cognee release.

## What a run does

1. **plan** reads every package and its version from the workspace
   (`scripts/packages.py`) and asks PyPI for `<distribution>/<version>`. A version
   that is already published is skipped under `all` and refused when the package
   was asked for by name.
2. The selected packages release in three tiers, because a connector depends on
   `cognee-connectors-core` from PyPI and the Google family also depends on
   `cognee-connector-google`:
   1. `cognee-connectors-core`,
   2. `cognee-connector-google`,
   3. every other connector, in parallel.

   A tier starts only when the one before it did not fail, so no connector is
   published while the core version it needs is missing. Within tier 3 one failed
   connector does not cancel the others.
3. Per package: `uv build --package <distribution>`, a check that the sdist and
   wheel carry the version, a SLSA build-provenance attestation, the upload to PyPI
   through Trusted Publishing (with PEP 740 attestations), and only then the tag
   `<distribution>-v<version>`. A failed upload leaves no tag, so a rerun retries
   the same version. There is no GitHub Release for connector packages, as for
   cognee-mcp.

## Prepare a release

1. Bump the version in the package's `pyproject.toml`
   (`connectors/<p>/pyproject.toml` or `core/pyproject.toml`). Versions are PEP
   440; `scripts/check_pins.py` checks them in CI. `cognee-connectors-core` follows
   semver; connectors pin `cognee-connectors-core>=0.1,<1` until core reaches 1.0
   (docs/ARCHITECTURE.md, rule 7). A connector that needs a newer core raises its
   lower bound, and both are bumped in the same PR.
2. Run `uv lock` in `cognee-connectors/`.
3. Open a PR to `dev` and merge it; the bump ships with the next stable cognee
   release, or run `release_connectors.yml` on `main` once it is there.

Dev releases of a connector are cut from `dev` with a pre-release version
(`0.2.0.dev1`) by running the workflow by hand on `dev`.

## Trusted Publishing setup

Publishing uses PyPI Trusted Publishing (OpenID Connect): no API token, no
secret. Every package has its own GitHub environment, `pypi-<distribution>`
(for example `pypi-cognee-connector-notion`); the release jobs declare it, and
GitHub creates it on first use. Add required reviewers to one of them to gate
that package's uploads.

The PyPI publisher of each package:

| Field | Value |
| --- | --- |
| PyPI project name | the distribution name, e.g. `cognee-connector-notion` |
| Owner | `topoteretes` |
| Repository name | `cognee` |
| Workflow name | `release_connectors.yml` |
| Environment name | `pypi-<distribution>`, e.g. `pypi-cognee-connector-notion` |

The environment must match on both sides: a publisher registered without one, or
with another name, fails the upload with `invalid-publisher`.

- A package already on PyPI gets the publisher under the project's
  *Settings → Publishing*.
- A package that was never uploaded has no project yet, so it needs a **pending
  publisher** on the [account publishing page](https://pypi.org/manage/account/publishing/),
  which creates the project on the first upload. PyPI refuses two pending
  publishers with the same repository, workflow and environment (hence one
  environment per package) and allows **at most 3 pending publishers per account
  at a time**. A pending publisher turns into a normal one, and frees its slot,
  when its first upload creates the project.

### First release of the initial packages

Because of the three-slot limit the thirteen initial packages go out in waves.
Each wave: register up to three pending publishers, then run the workflow for
those packages (or `all`: the packages without a publisher fail at the upload,
leave no tag, and go in a later wave).

| Wave | Packages |
| --- | --- |
| 1 | `cognee-connectors-core`, `cognee-connector-apollo`, `cognee-connector-asana` (registered) |
| 2 | `cognee-connector-google`, `cognee-connector-github`, `cognee-connector-linear` |
| 3 | `cognee-connector-gmail`, `cognee-connector-google-calendar`, `cognee-connector-google-drive` |
| 4 | `cognee-connector-notion`, `cognee-connector-obsidian`, `cognee-connector-onenote` |
| 5 | `cognee-connector-slack` |

Core goes first because every connector needs it on PyPI; Google goes before the
Google family for the same reason.

## Release validation

1. The package is on PyPI: `https://pypi.org/project/<distribution>/<version>/`,
   with a *Provenance* section on the file pages.
2. The tag `<distribution>-v<version>` exists.
3. Verify the provenance of a downloaded wheel if you want to:
   `gh attestation verify ./<module>-<version>-py3-none-any.whl --repo topoteretes/cognee`.
4. A fresh `pip install <distribution>` imports the package, and
   `pip install "<distribution>[cognee]"` installs the host adapter
   (`python scripts/check_entry_points.py <distribution>` resolves its entry points).

## Adding a package

A new connector needs, besides its directory and inventory entry
(`scripts/new_connector.py`), an entry in the `connector` choice of
`release_connectors.yml` (CI fails without it) and its PyPI publisher, a pending
one before its first release.
