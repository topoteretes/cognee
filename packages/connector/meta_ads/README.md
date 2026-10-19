# Meta Ads connector

This package wraps dlt's verified Facebook Ads source for read-only ingestion of Meta campaigns, ad sets, ads, creatives, and ad-level insights. Instagram placements use the same Marketing API. Each row is rendered into the title/content columns cognee's document ingestion expects, with Meta IDs and parent IDs retained as metadata columns.

## Install and credentials

```bash
pip install -e '.[test]'
dlt init facebook_ads duckdb
```

Create a Meta app, add the ads_read permission, and generate a long-lived user or system-user access token with access to the selected ad accounts. Keep the token outside source control:

```bash
set META_ADS_ACCESS_TOKEN=your-token
set META_ADS_ACCOUNT_IDS=act_123456789,act_987654321
```

The dlt verified-source initializer creates the facebook_ads source module in the working directory; run the connector from that environment so it can import the generated module.

The token is passed to the dlt source and is not included in emitted rows. Rotate it through Meta's tools when it expires; token refresh is not included.

## Use with cognee

```python
import os

import cognee
from cognee_community_connector_meta_ads import meta_ads_source

source = meta_ads_source(
    access_token=os.environ["META_ADS_ACCESS_TOKEN"],
    account_ids=os.environ["META_ADS_ACCOUNT_IDS"].split(","),
    window="last_30d",
)

await cognee.remember(
    source,
    dataset_name="meta_ads",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

The window accepts relative periods such as last_30d or last_90d. Reuse the same dlt pipeline state and dataset on later syncs.

## API version and sync behavior

Graph API v23.0 is pinned in settings.py. Check Meta's Marketing API support before upgrading, update the constant, and review the requested fields.

dlt's verified object resources use cursor paging, and this wrapper merges by Meta object ID. The verified source currently exposes its incremental cursor for insights dates, not updated_time cursors for campaign, ad set, ad, or creative resources. Changed-only object fetches therefore are not guaranteed. Insights are aggregated without daily breakdowns.

The verified source does not expose response-header hooks for Meta's business-use-case quota header or custom Graph API retries. It also does not provide an authoritative full-ID listing or graph deletion callback. reconcile_ids is a safe helper for a caller that has such a listing: it applies no deletion if listing fails partway through. Archived and deleted states are requested where supported and remain in the merged source data.

## Runnable example

Set META_ADS_ACCESS_TOKEN and META_ADS_ACCOUNT_IDS, then run from this directory:

```bash
python examples/example.py
```

The example performs a real sync and needs a cognee LLM configuration. Tests use local fixtures only.
