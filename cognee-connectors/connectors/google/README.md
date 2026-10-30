# cognee-connector-google

The Google OAuth code shared by cognee's Google Drive, Gmail and Google Calendar
connectors. It syncs nothing itself: `cognee-connector-google-drive`, `cognee-connector-gmail`
and `cognee-connector-google-calendar` depend on it for the log-safe OAuth client
(`exchange_code`, `refresh_access_token`, `revoke_token`, `fetch_userinfo`), the settings
base their `GOOGLE_<PRODUCT>_*` settings subclass (`GoogleSettings`), the connector
contract they derive from (`google_descriptor`, `GoogleConnector`) and, with the `cognee`
extra, the `GoogleOAuthIntegration` base their cognee host adapters subclass. Install one
of those packages to sync data; this one comes with it.

Every Google product connects one Google account, personal or Workspace, through an ordinary
per-user consent at the same endpoints. Google's token response carries no account identity,
so the fresh access token is spent on one userinfo call and the credential is keyed on the
stable Google subject (`userinfo.sub`); the account's email and hosted domain are kept as
cleartext metadata and a Workspace account is told from a personal one by the hosted-domain
claim. Two parameters on the authorize URL carry the design: `access_type=offline` with
`prompt=consent` is what makes Google issue a refresh token, and `include_granted_scopes`
is deliberately absent, so that disconnecting one product never revokes another's grant.

## Install

```bash
pip install cognee-connector-google             # the OAuth client, settings base and contract
pip install "cognee-connector-google[cognee]"   # plus the cognee host base (installs cognee)
```

The package advertises `CONNECTOR` under the `cognee.connectors` entry point as `google`: the
identity-only base connector (scopes `openid email`), built without settings. A host that
registers it gets a provider that connects a Google account and syncs nothing; the data
connectors are the three service packages.

## Configuration

`cognee_connector_google.settings.GoogleSettings` is a base class, not a configured
provider: it adds `state_secret` to the shared `client_id`, `client_secret`, `redirect_uri`
and `frontend_base_url` fields and sets no environment prefix of its own. Each product
reads the same five fields under its own prefix, exactly as in cognee:

| Field | Google Drive | Gmail | Google Calendar |
| --- | --- | --- | --- |
| `client_id` | `GOOGLE_DRIVE_CLIENT_ID` | `GOOGLE_GMAIL_CLIENT_ID` | `GOOGLE_CALENDAR_CLIENT_ID` |
| `client_secret` | `GOOGLE_DRIVE_CLIENT_SECRET` | `GOOGLE_GMAIL_CLIENT_SECRET` | `GOOGLE_CALENDAR_CLIENT_SECRET` |
| `redirect_uri` | `GOOGLE_DRIVE_REDIRECT_URI` | `GOOGLE_GMAIL_REDIRECT_URI` | `GOOGLE_CALENDAR_REDIRECT_URI` |
| `frontend_base_url` | `GOOGLE_DRIVE_FRONTEND_BASE_URL` | `GOOGLE_GMAIL_FRONTEND_BASE_URL` | `GOOGLE_CALENDAR_FRONTEND_BASE_URL` |
| `state_secret` | `GOOGLE_DRIVE_STATE_SECRET` | `GOOGLE_GMAIL_STATE_SECRET` | `GOOGLE_CALENDAR_STATE_SECRET` |

The products share the Google Cloud project, not the configuration block: each has its own
OAuth client and its own consent screen (see the product READMEs). This package reads no
environment variable itself.

## Usage

The service packages are the ones to use. What this package offers them, and any other
Google product a host wants to add on the same contract:

```python
from cognee_connector_google import GoogleConnector, GoogleSettings, google_descriptor
from pydantic_settings import SettingsConfigDict


class GoogleTasksSettings(GoogleSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_TASKS_", extra="ignore")


DESCRIPTOR = google_descriptor(
    "google_tasks", "Google Tasks", "openid email https://www.googleapis.com/auth/tasks.readonly"
)
CONNECTOR = GoogleConnector(DESCRIPTOR, settings=GoogleTasksSettings())
```

`google_descriptor` writes Google's OAuth shape down once (the authorize and token endpoints,
the `response_type=code`, `access_type=offline`, `prompt=consent` parameters, the paths to
the account under `userinfo`, an hourly poll interval since Google sends no webhooks on this
path). `GoogleConnector` adds the userinfo lookup (`resolve_account`), the account-type
metadata, the refresh grant (Google issues no new refresh token; the stored one is kept) and
a best-effort revoke that never raises. The OAuth client the cognee host adapter uses is
exported too:

```python
from cognee_connector_google import (
    exchange_code,
    fetch_userinfo,
    refresh_access_token,
    revoke_token,
)
```

Its errors are `GoogleAuthError(operation, code)`, with a stable, log-safe code
(`invalid_grant`, `http_502`, `no_access_token`) and never a response body.

## Running on a cognee server

There is nothing to register for this package: Google itself is not a provider in cognee's
registry. `cognee_connector_google.cognee_host.oauth_integration.GoogleOAuthIntegration` is
the base the Drive, Gmail and Calendar adapters subclass (consent URL, code exchange plus
userinfo, installation parsing, refresh five minutes before expiry, revoke on
`invalid_grant`, best-effort remote revoke), and `cognee_connector_google.cognee_host.ingestion`
is the Google-named alias of `cognee.modules.integrations.ingestion` that the Cloud pod
imports. Register the service packages instead:

```python
from cognee_connector_gmail.cognee_host import register as register_gmail
from cognee_connector_google_calendar.cognee_host import register as register_calendar
from cognee_connector_google_drive.cognee_host import register as register_drive

register_drive()
register_gmail()
register_calendar()
```

## What is synced

Nothing: this package carries no DLT source. See `cognee-connector-google-drive`,
`cognee-connector-gmail` and `cognee-connector-google-calendar`.

## Development

From the repository root, with the workspace's own interpreter (`uv sync --all-packages --group dev`):

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/google python -m pytest connectors/google/tests -q
python scripts/check_cognee_free.py cognee_connector_google
```

`tests/unit` covers the contract object (`test_connector.py`, no cognee needed), the OAuth
client's log-safe errors, and the cognee host base shared by the three products. The tests
that drive that base through the three adapters (`test_google_oauth_adapters.py`,
`test_google_credential_lifecycle.py`, `test_google_dataset_summary.py`) need the service
packages importable as well; they skip when run with this package alone and run in full with
`PYTHONPATH=core:connectors/google:connectors/google-drive:connectors/gmail:connectors/google-calendar`
or a synced workspace.
