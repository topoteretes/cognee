# Security Policy

The Cognee team takes security issues seriously. This page says how to report a
vulnerability, what happens after you do, how fast we commit to fixing it, and
how to find out about security releases if you run Cognee yourself.

## Supported versions

Security fixes ship in the latest release on [PyPI](https://pypi.org/project/cognee/),
[Docker Hub](https://hub.docker.com/r/cognee/cognee) and
[GitHub Releases](https://github.com/topoteretes/cognee/releases). We do not
backport fixes to older minor versions. Keep your deployment on the latest
release to receive security fixes.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| Earlier releases | No, upgrade to the latest release |

Python 3.10 through 3.14 are supported, as declared in `pyproject.toml`.

## Reporting a vulnerability

Please do not open a public GitHub issue for a security problem.

Report it through either channel:

- Email [security@cognee.ai](mailto:security@cognee.ai) with the word
  **SECURITY** in the subject line.
- Use GitHub's private vulnerability reporting:
  [Report a vulnerability](https://github.com/topoteretes/cognee/security/advisories/new).

Include what you can of the following. It shortens triage and gets you a
faster answer:

- The affected component (API route, CLI command, MCP tool, loader, database
  adapter, …) and the Cognee version or commit.
- Steps to reproduce, a proof of concept, or the request and response that
  shows the problem.
- The impact you believe it has, and the configuration it needs
  (for example `ENABLE_BACKEND_ACCESS_CONTROL` on or off, which graph or
  vector backend).
- How you would like to be credited, if at all.

We accept reports in English. If you want to encrypt your report, ask for a
key in a first plain email and we will send one.

### Safe harbour

We will not pursue legal action against researchers who report in good faith
and who make a reasonable effort to avoid privacy violations, data
destruction, and service disruption, who only test against systems they own
or have permission to test, and who give us the time set out below to fix the
issue before disclosing it.

## What happens after you report

| Step | Target |
| --- | --- |
| Acknowledgement that we received the report | 2 business days |
| Triage: confirmed or not, and an initial severity | 5 business days |
| Status update while a fix is in progress | at least every 10 business days |
| Credit in the advisory, if you want it | at publication |

Severity is assessed with CVSS v3.1. Where the score depends on deployment
assumptions we rate against the default configuration documented in
`CLAUDE.md` and `.env.template`.

## Remediation windows

The windows below are our commitment for a confirmed vulnerability in
Cognee's own code, counted from the day the report is confirmed. A fix means a
release that resolves the issue or, where a release is not yet possible, a
documented mitigation that removes the exposure.

| Severity | CVSS v3.1 | Fix released within |
| --- | --- | --- |
| Critical | 9.0 – 10.0 | 7 days |
| High | 7.0 – 8.9 | 30 days |
| Medium | 4.0 – 6.9 | 60 days |
| Low | 0.1 – 3.9 | 90 days |

Ninety days is the ceiling for any severity. If a fix needs more time, for
example because it depends on an upstream project, we say so in the advisory
and publish the mitigation at the 90-day mark.

Vulnerabilities in third-party dependencies are handled by upgrading the
dependency. Dependabot and the nightly container scan raise these
automatically. They follow the same windows, measured from when a fixed
upstream version is available.

## Disclosure

We practise coordinated disclosure.

- A confirmed vulnerability gets a
  [GitHub Security Advisory](https://github.com/topoteretes/cognee/security/advisories)
  with a CVE identifier requested through GitHub.
- The advisory is published when the fixed release is out, and no later than
  90 days after confirmation, whichever comes first.
- The release notes of the fixing release carry a **Security** section that
  names the advisory.
- Reporters are credited in the advisory unless they ask not to be.

If you believe a vulnerability is being exploited in the wild, say so in your
report. We prioritise those above the windows above.

## How to get notified of security releases

Self-hosting users are responsible for upgrading. These are the channels we
publish on, in the order they fire:

1. **GitHub Security Advisories.** Watch the repository with
   *Watch → Custom → Security alerts* to receive an email for each advisory.
   Published advisories also feed the GitHub Advisory Database, so Dependabot
   and `pip-audit` flag the vulnerable version in your own project.
2. **GitHub Releases.** Each fixing release is tagged on
   [the releases page](https://github.com/topoteretes/cognee/releases) with a
   Security section in its notes. Subscribe to releases with
   *Watch → Custom → Releases*.
3. **Discord.** Releases are announced in the
   [Cognee Discord](https://discord.gg/NQPKmU5CCg) release channel.
4. **Direct notification.** Customers with a support or enterprise agreement
   are notified by email at the contact on their agreement, before or at
   publication, as set out in that agreement.

## Scope

In scope:

- The `cognee` Python package and everything under `cognee/` in this
  repository, including the HTTP API, CLI, MCP server, loaders, pipelines and
  the in-tree database adapters.
- The official Docker images built from this repository.
- The release process: PyPI and Docker Hub publishing, build provenance and
  SBOM attestations.

Out of scope, please report to the owner instead:

- Vulnerabilities in LLM, embedding or database providers that Cognee
  connects to. Cognee sends chunk text and queries to the providers you
  configure; their handling of that data is governed by their terms.
- The infrastructure, operating system and network of a self-hosted
  deployment.
- Community adapters and integrations published from
  [cognee-community](https://github.com/topoteretes/cognee-community); report
  those on that repository.
- Findings that require a deliberately insecure configuration that the
  documentation already warns about, such as `ENABLE_BACKEND_ACCESS_CONTROL=false`
  on a network-exposed server, or raw Cypher left enabled for untrusted users.
  We still welcome these as hardening suggestions in a normal issue.
- Reports from automated scanners with no demonstrated impact.

## Automated assurance

Independently of reports, the repository runs:

- [OpenSSF Scorecard](https://github.com/ossf/scorecard) on the default
  branch (`.github/workflows/scorecard.yml`).
- Dependabot security updates and GitHub secret scanning.
- A nightly Trivy scan of the published Docker images
  (`.github/workflows/docker_validation_nightly.yml`).
- PEP 740 provenance and SLSA build-provenance attestations on every release
  (`.github/workflows/release.yml`), so a published artifact can be verified
  against the commit that built it.

## Questions

For anything about this policy that is not a vulnerability report, open a
discussion on GitHub or ask in Discord. For contractual security questions,
contact your Cognee account contact.
