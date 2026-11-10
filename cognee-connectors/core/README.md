# cognee-connectors-core

The contract every cognee connector implements, with no host in it. A connector
package (`cognee-connector-<provider>`) depends on this package and on its vendor
SDK; the hosts — cognee's own server and SDK, cloud-backend — depend on it to find
and drive the connectors. `cognee` itself is never a runtime dependency of this
package: everything here imports with cognee absent.

```bash
pip install cognee-connectors-core          # the contract
pip install "cognee-connectors-core[dlt]"   # plus the dlt range connectors with a DLT source pin
```

Python 3.10 – 3.14. Apache-2.0.

## Modules

| Module | What it owns |
| --- | --- |
| `cognee_connectors_core.base` | `Connector` (the abstract contract), `ConnectorInstallation`, `SyncScope`, `ConnectorResource`, `WebhookDispatch`, `RefreshedTokens`, `WebhookVerifier`, the `Credential` protocol, `account_slug`, `dataset_name_for` |
| `cognee_connectors_core.descriptor` | `ConnectorDescriptor`, a Tier 1 connector expressed as data, plus `resolve_path` and `payload_paths` |
| `cognee_connectors_core.oauth2` | `DescriptorConnector`, which turns a descriptor into a working connector: authorize URL, code exchange, installation parsing, refresh, webhook verification and handling, and the `resolve_account` hook for Tier 2 |
| `cognee_connectors_core.signatures` | The webhook HMAC schemes (`SHA256_PREFIX`, `BARE_HEX`, `SLACK_V0`, `TIMESTAMPED`) and `HmacVerifier` |
| `cognee_connectors_core.oauth_state` | Provider-agnostic OAuth `state` minting and validation (`make_state`, `validate_state`) |
| `cognee_connectors_core.settings` | `ConnectorSettings`, the pydantic-settings base every provider settings class subclasses, and `require` |
| `cognee_connectors_core.dlt_conventions` | The attribute and column names cognee's ingestion reads off a DLT source (`cognee_document_source`, `cognee_pipeline_scope`, `cognee_node_set`, `cognee_structure`) and the helpers around them |
| `cognee_connectors_core.registry` | `use_connector`, `get_connector`, `find_connector`, `catalog()` and the entry-point loaders |
| `cognee_connectors_core.http` | The two-method `HttpClient` protocol and its aiohttp default |
| `cognee_connectors_core.errors` | `ConnectorError`, `WebhookVerificationError`, `RateLimitedError` |
| `cognee_connectors_core.testing` | `FakeHttpClient`, `FakeSettings`, `FakeCredential`, `sign_body`, `webhook_headers` |

The design, the three tiers and the rules every package follows are in
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) at the repo root.

## How a host uses the loaders

Connector packages advertise themselves through three entry-point groups; a host
never imports a provider by name.

```python
from cognee_connectors_core import registry

# Group "cognee.connectors": {provider: Connector}, each one registered.
connectors = registry.load_connectors()
# Group "cognee.sources": {name: source factory}; nothing is called.
sources = registry.load_sources()
# Group "cognee.integrations": calls each package's register(); the names that ran.
registered = registry.load_host_registrations()

# KeyError for an unknown provider, which a host answers with 404.
connector = registry.get_connector("notion")
# provider, display_name, auth, tier, has_webhooks, supports_resource_selection, ...
for entry in registry.catalog():
    print(entry["provider"], entry["auth"], entry["has_webhooks"])
```

Each entry point is loaded on its own: a package that fails to import is logged
and skipped, so one broken connector cannot hide the others. A host that already
knows its connectors registers them directly with `registry.use_connector(CONNECTOR)`.

The host owns everything the contract leaves out: credential storage and
encryption, the OAuth routes (minting `state` with `oauth_state.make_state` and
checking it with `validate_state`), webhook delivery dedupe (claim
`WebhookDispatch.delivery_id`, or `connector.webhook_delivery_key(body, headers)`
before calling `handle_webhook`), and running a sync. Token material reaches a
connector only as the decrypted `tokens` mapping the host passes in.

## How a connector author uses `DescriptorConnector`

A Tier 1 provider is a descriptor, a settings class and nothing else:

```python
from pydantic_settings import SettingsConfigDict

from cognee_connectors_core import (
    ConnectorDescriptor,
    ConnectorSettings,
    DescriptorConnector,
    SignatureScheme,
)


class LinearSettings(ConnectorSettings):
    model_config = SettingsConfigDict(env_prefix="LINEAR_", extra="ignore")
    webhook_secret: str = ""  # LINEAR_WEBHOOK_SECRET


DESCRIPTOR = ConnectorDescriptor(
    provider="linear",
    display_name="Linear",
    authorize_url="https://linear.app/oauth/authorize",
    token_url="https://api.linear.app/oauth/token",
    scopes="read",
    authorize_params=(("response_type", "code"), ("actor", "app")),
    account_id_path="organization.id",
    account_label_path="organization.name",
    signature_scheme=SignatureScheme.BARE_HEX,
    signature_header="linear-signature",
    webhook_account_id_path="organizationId",
    webhook_delivery_id_path="webhookId",
)

CONNECTOR = DescriptorConnector(DESCRIPTOR, LinearSettings())
```

`DescriptorConnector` then implements the whole `Connector` contract:
`authorize_url(state)`, `exchange_callback(code, params)`,
`parse_installation(account_data)`, `refresh(credential, tokens)`,
`webhook_verifier()`, `handle_webhook(raw_body, headers)`,
`webhook_delivery_key(raw_body, headers)`, `sync_scope(credential, tokens,
resources=...)`, `dataset_name(credential)`, polling and `content_type`.
Settings are read at call time, so a value rotated on the settings object takes
effect on the next request; a missing one raises `RuntimeError` naming the
environment variable (`LINEAR_CLIENT_ID is not configured`).

A Tier 2 provider, whose token response does not say which account authorized,
subclasses it and overrides one hook:

```python
class LinearConnector(DescriptorConnector):
    async def resolve_account(self, token_response):
        status, body = await self.http.get_json(
            "https://api.linear.app/graphql?query=...",
            {"Authorization": f"Bearer {token_response['access_token']}"},
        )
        # Merged over the token response; account_id_path then reads organization.id.
        return {"organization": body["data"]["viewer"]["organization"]}
```

An API-key or local provider is a descriptor with `auth="api_key"` or
`auth="local"`; the OAuth fields may stay empty and the OAuth members raise
`NotImplementedError`, while the provider still has a name, a dataset, a
content type and a catalog entry. A provider whose auth model is not OAuth2 at all
(GitHub Apps) subclasses `Connector` directly.

Webhook verification is a scheme, not code: `HmacVerifier` verifies over the raw
bytes, in constant time, and refuses with `WebhookVerificationError` carrying the
status a host should answer with (`401`, `413` over `MAX_BODY_BYTES`). Header
names are matched case-insensitively.

### Testing a connector

```python
from cognee_connectors_core.testing import FakeHttpClient, FakeSettings, webhook_headers

token_response = {"access_token": "tok", "organization": {"id": "org-1"}}
http = FakeHttpClient({("POST", DESCRIPTOR.token_url): (200, token_response)})
connector = DescriptorConnector(DESCRIPTOR, FakeSettings(webhook_secret="s3"), http)

account_data = await connector.exchange_callback("code", {})
installation = connector.parse_installation(account_data)
assert installation.provider_account_id == "org-1"
assert http.last_call.data["grant_type"] == "authorization_code"

body = b'{"organizationId": "org-1", "webhookId": "d1"}'
headers = webhook_headers(DESCRIPTOR, "s3", body)
verified = await connector.webhook_verifier().verify(body, headers)
dispatch = await connector.handle_webhook(verified, {})
assert dispatch.provider_account_id == "org-1" and dispatch.delivery_id == "d1"
```

## DLT conventions

A DLT source opts into cognee's document ingestion by setting the attributes
named in `dlt_conventions` (`DOCUMENT_SOURCE_ATTR`, `PIPELINE_SCOPE_ATTR`) and
by writing the `NODE_SET_COLUMN` / `STRUCTURE_COLUMN` columns with
`DOCUMENT_COLUMN_HINTS`. The values are byte-identical to cognee's
`cognee.tasks.ingestion.dlt_utils`; connectors import them from here so no
package needs cognee at runtime.

One value is a version, not a name. `DOCUMENT_SYNC_VERSION` is the protocol
version cognee's ingestion implements and it moves with cognee releases; the
copy here is the version this package was cut at. A source that needs a
capability (per-row node sets from version 2, structure from version 3) gates
on the **host's** value, never on the constant:

```python
from cognee_connectors_core.dlt_conventions import require_document_sync_version

# At the top of the source factory, before any provider call.
require_document_sync_version(3, provider="Notion", feature="reads per-row node sets and structure")
```

`host_document_sync_version()` is what it reads: the installed cognee's value,
`0` for a cognee without DLT document ingestion, and this package's own value
only when cognee is not installed at all (nothing can ingest then, and a
connector's tests run without it).

## Development

From the repo root, with a Python that has pytest and pytest-asyncio:

```bash
PYTHONPATH=core python -m pytest core/tests -q
ruff check core && ruff format --check core
```

The suite is pure: no network, no database, no cognee. One test compares the
DLT conventions against cognee's own module when cognee happens to be
importable, and skips otherwise.
