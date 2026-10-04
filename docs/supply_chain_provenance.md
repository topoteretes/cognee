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

The relevant workflows are `.github/workflows/release.yml` (tagged library releases),
`.github/workflows/dev_canary_release.yml` (weekly dev canaries), and
`.github/workflows/release_mcp.yml` (manual MCP releases).

---

## One-time setup: PyPI Trusted Publishing

PyPI only accepts and displays PEP 740 attestations when a package is uploaded
through **Trusted Publishing** (OpenID Connect), not an API token. The library
workflows have already been switched to OIDC (`id-token: write`, no
`UV_PUBLISH_TOKEN`), but a project owner must register the trusted publishers
on PyPI **once**. `release_mcp.yml` is the exception until its publisher exists:
see [cognee-mcp still uploads with a token](#cognee-mcp-still-uploads-with-a-token).

PyPI scopes trusted publishers per project. Register **three** publishers, one
per workflow: two on `cognee` and one on the separate `cognee-mcp` project.

1. Open the publishing settings for [cognee](https://pypi.org/manage/project/cognee/settings/publishing/)
   and [cognee-mcp](https://pypi.org/manage/project/cognee-mcp/settings/publishing/).
2. Under **Add a new publisher** → **GitHub**, add each publisher to the
   PyPI project listed below:

   | Field | Release publisher | Canary publisher | MCP release publisher |
   | --- | --- | --- | --- |
   | PyPI project | `cognee` | `cognee` | `cognee-mcp` |
   | Owner | `topoteretes` | `topoteretes` | `topoteretes` |
   | Repository | `cognee` | `cognee` | `cognee` |
   | Workflow name | `release.yml` | `dev_canary_release.yml` | `release_mcp.yml` |
   | Environment | *(leave blank)* | *(leave blank)* | *(leave blank)* |

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

3. Save all three publishers.

Release MCP by running `release_mcp.yml` from the `main` branch in the Actions
tab. Other refs fail explicitly. The workflow reads the version from
`cognee-mcp/pyproject.toml`, refuses to run if that version is already on PyPI,
uploads it, and tags the commit `cognee-mcp-v<version>` (its own namespace,
since cognee-mcp versions independently of the library).

After the publishers are registered, the next library release uploads with
provenance automatically.

> ⚠️ **Do not run a library release before its publishers are registered** — the
> publish step will fail OIDC auth. The release workflow is
> `workflow_dispatch`-only, so you control the timing.

### cognee-mcp still uploads with a token

The `cognee-mcp` publisher in the table above has not been registered, and only
the owner of that PyPI project can add it. Until then `release_mcp.yml` uploads
with the `PYPI_TOKEN` repository secret, the account-wide token the library used
before it moved to OIDC. Two consequences:

- `cognee-mcp` files on PyPI carry no PEP 740 attestations. That includes 0.5.6,
  which was uploaded by hand with the same token on 2026-10-01 after the first
  OIDC run failed with `invalid-publisher`. The SLSA build provenance on GitHub
  is still produced for every workflow release.
- The token can publish every project its account owns, `cognee` included, so
  it is a broader credential than this workflow needs.

Moving over needs no workflow change, because the publish step uses Trusted
Publishing whenever the secret is absent. Do it in this order:

1. Register the `cognee-mcp` publisher from the table above.
2. Delete the `PYPI_TOKEN` secret.

The order matters. Deleting the secret first leaves MCP releases with no way to
authenticate.

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
