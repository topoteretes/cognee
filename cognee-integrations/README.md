# Cognee integrations

Plugins that give agent tools persistent memory backed by Cognee. Each plugin
is its own package, released on its own schedule.

The integrations are moving here from
[topoteretes/cognee-integrations](https://github.com/topoteretes/cognee-integrations),
one at a time. Until an integration is listed below, its source, docs and
releases are still in that repository.

## Moved so far

| Integration | Package | Source |
|---|---|---|
| Hermes Agent | [`cognee-integration-hermes-agent`](https://pypi.org/project/cognee-integration-hermes-agent/) | [`integrations/hermes-agent`](integrations/hermes-agent) |

## Layout

```
cognee-integrations/
├── pyproject.toml   Shared lint and test settings for every plugin below
└── integrations/
    ├── conformance/ Case tables that every plugin tests against
    └── <plugin>/    One folder per plugin, with its own pyproject.toml or
                     package.json, tests, README and CHANGELOG
```

The layout is the same as in topoteretes/cognee-integrations, with this
folder in place of that repository's root, so paths inside a plugin keep
working when it moves. Everything at that repository's root has a place here:

| In topoteretes/cognee-integrations | Here |
|---|---|
| `integrations/<plugin>/` | `integrations/<plugin>/` |
| `integrations/tests/` (the shared hook-plugin suite) | `integrations/tests/` |
| `integrations/inventory.yml`, `integrations/CONFIGURATION.md` | `integrations/` |
| `scripts/`, `assets/`, `n8n_workflows/` | `scripts/`, `assets/`, `n8n_workflows/` |
| `.github/workflows/` | the shared workflows in this repository's `.github/workflows/` |
| `reports/` | not moved |
| `.claude-plugin/`, `.cursor-plugin/`, `.grok-plugin/`, `.agents/plugins/` | not moved; see [Marketplace plugins](#marketplace-plugins) |

`integrations/conformance/` is a copy. Two of its tables are generated from
the Claude Code and Codex hook plugins, which are still in
topoteretes/cognee-integrations, so that copy is canonical until they move
here. `test_integrations.yml` fails when this copy differs from it.

## Marketplace plugins

The Claude Code, Codex, Cursor and Grok Bot plugins are installed from agent
marketplaces, not from a package registry. Their code moves here like the rest.
They are distributed through the official listings (the claude.ai Directory,
and each agent's own marketplace), pointed at this repository. Nothing is added
at this repository's root for them.

topoteretes/cognee-integrations keeps only its marketplace manifests, repointed
here, for users who added that repository as a marketplace by hand. Once they
have moved to the official listings, it is archived.

## Moving a plugin

1. Copy the plugin folder from topoteretes/cognee-integrations to
   `integrations/<plugin>/` and add it to the table above.
2. If it is listed in a marketplace or directory, point that listing at
   `topoteretes/cognee` and `cognee-integrations/integrations/<plugin>` first.
   The Directory reviews each version, so allow time for that.
3. In topoteretes/cognee-integrations, in the same change: replace the plugin
   folder with a note pointing here, remove it from that repository's CI and
   release workflows, and mark it moved in `integrations/inventory.yml`.
4. Triage the CodeQL alerts the move raises on the copied code, and dismiss
   the false positives with a reason.

## Development and releases

Each plugin's README covers its own setup, tests and release steps.
`.github/workflows/test_integrations.yml` lints and tests every plugin under
`integrations/` that has a `pyproject.toml`, and runs each plugin's live round
trip (`tests/test_integration_*.py`) nightly against `dev`, posting every
result to `#nightly-ci-integrations` on Slack. It needs no edit when a plugin
moves here. Releases are per plugin for now, for example
`release_hermes_plugin.yml` for Hermes.
