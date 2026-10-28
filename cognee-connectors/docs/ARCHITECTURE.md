# cognee-connectors architecture

One repo, one contract, one package per data source. cognee is a development
dependency: it is installed to run the end-to-end tests, never required at
runtime by a connector package.

## Layers

A connector splits into three layers. Only the third one knows a host.

| Layer | Contents | Depends on | Lives in |
| --- | --- | --- | --- |
| Source | Provider client and the DLT source: pagination, incremental cursors, forget-on-delete, records rendered as documents | `cognee-connectors-core[dlt]`, the vendor SDK | `cognee_connector_<p>/source.py`, `client.py` |
| Connector | The provider's OAuth contract: a `ConnectorDescriptor` (tier 1 and 2) or a `Connector` subclass (tier 3), webhook verification, `list_resources`, `dataset_name` | `cognee-connectors-core` | `cognee_connector_<p>/connector.py`, `settings.py`, `webhooks.py` |
| Host binding | Credential storage, OAuth routes, running a sync, multi-tenancy | the host | cognee core (`cognee.modules.integrations`), cloud-backend, and the thin `cognee_connector_<p>/cognee_host/` adapter that plugs the connector into cognee's existing runtime |

`cognee-connectors-core` (module `cognee_connectors_core`) is the contract:

- `base`: `Connector`, `ConnectorInstallation`, `SyncScope`, `ConnectorResource`, `WebhookDispatch`, `WebhookVerifier`, `Credential` (a `Protocol` naming the credential fields a connector may read), `account_slug`, `dataset_name_for`.
- `descriptor`: `ConnectorDescriptor` and the dotted-path helpers (`resolve_path`, `payload_paths`).
- `oauth2`: `DescriptorConnector`, which turns a descriptor into a working connector: authorize URL, code exchange, installation parsing, refresh, webhook handling. HTTP goes through an injectable `HttpClient`, so tests never touch the network.
- `signatures`: the webhook HMAC schemes (`SHA256_PREFIX`, `BARE_HEX`, `SLACK_V0`, `TIMESTAMPED`) and `HmacVerifier`.
- `oauth_state`: provider-agnostic OAuth `state` signing and validation.
- `settings`: `ConnectorSettings`, the pydantic-settings base every provider settings class subclasses (`client_id`, `client_secret`, `redirect_uri`, `frontend_base_url`).
- `dlt_conventions`: the attribute and column names cognee's ingestion reads off a DLT source (`cognee_document_source`, `cognee_pipeline_scope`, `cognee_node_set`, `cognee_structure`) and the helpers around them. This is the one definition both cognee and the connectors use.
- `registry`: `use_connector`, `get_connector`, `catalog()`, and the entry-point loaders hosts call.
- `testing`: a scripted `FakeHttpClient` and fixtures for connector tests.

## Packages

```text
core/                                   cognee-connectors-core
connectors/<provider>/                  cognee-connector-<provider>
  cognee_connector_<provider>/
    __init__.py                         exports CONNECTOR and the *_source factories; imports no host
    source.py                           the DLT source
    client.py                           provider API client (when the provider has one)
    settings.py                         <PROVIDER>_* settings; env var names unchanged
    connector.py                        CONNECTOR: ConnectorDescriptor or Connector subclass
    webhooks.py                         webhook verification on the core contract (when the provider has webhooks)
    cognee_host/                        plugs into cognee 1.6.x; imports cognee; installed with the `cognee` extra
      __init__.py                       register(): registers the OAuthIntegration with cognee's registry
      adapter.py, sync.py, ...          the host-bound code moved from cognee.modules.integrations.<provider>
      routers.py                        routers(): [(APIRouter, prefix, tags)] when the provider has HTTP routes
  tests/unit/                           moved from cognee/tests/unit/modules/integrations
  tests/integration/                    moved from cognee/tests/integration/tasks/connectors (needs cognee)
  examples/                             runnable examples
  README.md                             install, configuration, usage, host registration
  pyproject.toml
```

Package names use dashes (`cognee-connector-google-drive`), module names underscores
(`cognee_connector_google_drive`). The shared Google OAuth client is its own package,
`cognee-connector-google`, that Drive, Gmail and Calendar depend on.

## Rules

1. **No runtime cognee dependency.** `cognee` never appears in a package's
   `[project.dependencies]`. Importing `cognee_connector_<p>` must succeed with cognee
   absent; `scripts/check_cognee_free.py` and the `clean-install` CI job enforce it.
   Host-bound code lives under `cognee_host/`, which the package `__init__` never
   imports, and is installed with the package's `cognee` extra
   (`pip install cognee-connector-<p>[cognee]`).
2. **DLT conventions come from the core package**, never from
   `cognee.tasks.ingestion.dlt_utils`. The values are identical; cognee's ingestion
   reads them by attribute name.
3. **Logging** uses `logging.getLogger(__name__)`.
4. **Settings** subclass `cognee_connectors_core.settings.ConnectorSettings` with the
   same `env_prefix` and field names the provider had in cognee. Environment variable
   names never change.
5. **Connectors own no relational tables.** Credentials live in the host's store.
6. **Entry points** are how hosts find a package. Three groups:
   - `cognee.connectors`: `<provider> = cognee_connector_<p>:CONNECTOR` (the contract object)
   - `cognee.sources`: `<source> = cognee_connector_<p>:<source>_source` (one per DLT source)
   - `cognee.integrations`: `<provider> = cognee_connector_<p>.cognee_host:register` (cognee 1.6.x registration; needs the `cognee` extra)
7. **Versions.** Every package has its own version. `cognee-connectors-core` follows
   semver; connectors pin `cognee-connectors-core>=0.1,<1` until core reaches 1.0.
   Connectors that ship a DLT source depend on `cognee-connectors-core[dlt]`, which is
   the one place the `dlt` range is pinned.
8. **Conventions as in cognee**: ruff (line length 100, double quotes), pre-commit,
   DCO, a Linear issue key in every internal PR title or branch, semantic PR titles,
   signed-off commits (`git commit -s`).

## How a host uses a package

- **An SDK user** installs the connector and hands cognee its source:
  `await cognee.remember(notion_source(...))`.
- **A cognee server (1.6.x)** installs `cognee-connector-<p>[cognee]` and calls
  `cognee_connector_<p>.cognee_host.register()` before the app starts serving and after
  `cognee.api.client` is imported: on a cognee that still bundles the provider's own
  module, that import registers the bundled adapter with a plain assignment, which
  replaces an earlier `register()`. The connector then appears in
  `GET /api/v1/integrations/status` and its OAuth flow runs on the generic
  `/api/v1/integrations/{provider}/...` routes. Providers with routes of their own
  (Slack) expose `routers()` for the host to mount.
  `cognee_connectors_core.registry.load_host_registrations()` does the same for every
  installed package at once.
- **A cognee release with the entry-point loader** (planned) needs no call: the
  server and the SDK discover installed packages through the entry points above.
- **cloud-backend** consumes `CONNECTOR` and the sources; its own credential store,
  fan-out and pod hop stay cloud-side.
