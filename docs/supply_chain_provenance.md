# Supply-chain provenance & release attestations

Cognee's release pipeline produces **verifiable provenance** for every shipped
artifact so consumers can independently confirm that a package on PyPI or an
image on Docker Hub was built from this repository by our CI — not tampered with
in transit or rebuilt by a third party.

This covers three layers of evidence:

| Artifact | Mechanism | Where it is recorded |
| --- | --- | --- |
| PyPI sdist + wheel | **PEP 740 digital attestations** via PyPI Trusted Publishing | PyPI project page ("Provenance" / "Attestations") |
| PyPI sdist + wheel | **SLSA build provenance** (`actions/attest-build-provenance`) | GitHub repo → *Attestations* tab |
| Docker images | **in-toto provenance + SBOM** (buildx `provenance`/`sbom`) | Pushed alongside the image manifest |

The relevant workflows are `.github/workflows/release.yml` (tagged releases)
and `.github/workflows/dev_canary_release.yml` (weekly dev canaries).

---

## One-time setup: PyPI Trusted Publishing

PyPI only accepts and displays PEP 740 attestations when a package is uploaded
through **Trusted Publishing** (OpenID Connect), not an API token. The release
workflows have already been switched to OIDC (`id-token: write`, no
`UV_PUBLISH_TOKEN`), but a project owner must register the trusted publishers
on PyPI **once**:

1. Go to <https://pypi.org/manage/project/cognee/settings/publishing/>.
2. Under **Add a new pending publisher** → **GitHub**, add **two** publishers
   (one per release workflow file):

   | Field | Release publisher | Canary publisher |
   | --- | --- | --- |
   | Owner | `topoteretes` | `topoteretes` |
   | Repository | `cognee` | `cognee` |
   | Workflow name | `release.yml` | `dev_canary_release.yml` |
   | Environment | *(leave blank)* | *(leave blank)* |

3. Save both.

> The **Environment** value must match the `environment:` declared on the
> publishing job. The workflows do not set one, so leave this blank — if you
> later add a GitHub Actions environment, set the same name on both sides or the
> OIDC publish step fails auth.

> **Optional hardening (not configured):** a GitHub Actions *environment* with
> required reviewers / branch restrictions can gate publishing so commit access
> alone does not grant PyPI publishing rights. To enable it, add
> `environment: <name>` to the publishing job and set the matching name on the
> PyPI publisher above. Note that required reviewers on the canary workflow
> would block its weekly cron.

After the publishers are registered, the next release uploads with provenance
automatically. The legacy `PYPI_TOKEN` secret can be removed once a release has
succeeded via Trusted Publishing.

> ⚠️ **Do not run a release before the publishers are registered** — the publish
> step will fail OIDC auth. The release workflow is `workflow_dispatch`-only, so
> you control the timing.

---

## Verifying provenance as a consumer

### PyPI package (PEP 740)

`pip` surfaces attestations from the PyPI "Provenance" section on the project /
file pages. You can also fetch the integrity/provenance metadata via the PyPI
JSON API:

```bash
curl -s https://pypi.org/pypi/cognee/json | jq '.urls[].provenance'
```

### PyPI package (SLSA, GitHub-hosted)

Download a wheel/sdist and verify the GitHub-hosted build provenance with the
GitHub CLI:

```bash
gh attestation verify ./cognee-<version>-py3-none-any.whl --repo topoteretes/cognee
```

A successful verification confirms the artifact's SHA-256 digest was produced by
a workflow in `topoteretes/cognee`.

### Docker image (in-toto provenance + SBOM)

```bash
# Provenance attestation
docker buildx imagetools inspect cognee/cognee:latest \
  --format '{{ json .Provenance }}'

# SBOM attestation
docker buildx imagetools inspect cognee/cognee:latest \
  --format '{{ json .SBOM }}'
```

---

## How this maps to trust signals

- **Package provenance** (HVTracker / supply-chain trackers): flips from *None*
  to *present* once Trusted Publishing uploads PEP 740 attestations.
- **OpenSSF Scorecard**
  - `Signed-Releases` — satisfied by attested PyPI artifacts.
  - `Token-Permissions` — release workflows declare a minimal top-level
    `permissions: contents: read` and opt into `id-token`/`attestations` only
    where needed.
  - `Pinned-Dependencies` — all actions in the release workflows are pinned to
    full commit SHAs (with a version comment).

When bumping a pinned action, update both the SHA and its trailing
`# vX.Y.Z` comment together.
