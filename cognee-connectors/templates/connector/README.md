# __DIST__

__DISPLAY_NAME__ connector for [cognee](https://github.com/topoteretes/cognee): a `dlt` source
that turns __DISPLAY_NAME__ records into documents in cognee memory, synced incrementally and
with forget-on-delete, plus the provider's connection contract (`CONNECTOR`) for a cognee server
or Cognee Cloud.

<!-- Replace the paragraph above with what the connector syncs, how (which records become
documents, how incremental syncs and deletions work) and what it leaves out. -->

## Install

```bash
pip install __DIST__             # the connector and its DLT source
pip install "__DIST__[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

## Configuration

Every environment variable the settings class (`__MODULE__.settings`) reads, with the
`__ENV_PREFIX__` prefix. Names never change between cognee and this package.

```bash
# -- __DISPLAY_NAME__ --------------------------------------------------------------
#__ENV_PREFIX__CLIENT_ID=""
#__ENV_PREFIX__CLIENT_SECRET=""
#__ENV_PREFIX__REDIRECT_URI="http://localhost:8000/api/v1/integrations/__PROVIDER__/callback"
#__ENV_PREFIX__FRONTEND_BASE_URL="http://localhost:3000"
```

## Usage

```python
import cognee
from __MODULE__ import __PROVIDER___source

await cognee.remember(
    __PROVIDER___source(token="..."),
    dataset_name="my___PROVIDER__",
    primary_key="id",
    write_disposition="merge",
)
```

## Running on a cognee server

```python
from __MODULE__.cognee_host import register

register()  # before the app starts; the provider appears in GET /api/v1/integrations/status
```

`routers()` in the same module lists the routers the provider mounts, if any.

## What is synced

<!-- The records that become documents, what is skipped, the limits of the provider's API. -->

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/__SLUG__ python -m pytest connectors/__SLUG__/tests -q
python scripts/check_cognee_free.py __MODULE__
```
