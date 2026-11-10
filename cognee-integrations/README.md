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
working when it moves.

`integrations/conformance/` is a copy. The plugins that have not moved yet
test against the copy in topoteretes/cognee-integrations, so a change to a
table has to be made in both places until they move.

## Development and releases

Each plugin's README covers its own setup, tests and release steps. CI for a
plugin lives in this repository's `.github/workflows/`, for example
`test_hermes_plugin.yml` and `release_hermes_plugin.yml` for Hermes.
