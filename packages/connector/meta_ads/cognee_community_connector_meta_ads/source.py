from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .settings import (
    ACTIVE_AND_ARCHIVED_STATES,
    AD_FIELDS,
    AD_SET_FIELDS,
    CAMPAIGN_FIELDS,
    CREATIVE_FIELDS,
    GRAPH_API_VERSION,
    INSIGHT_FIELDS,
)
from .transform import render_ad, render_ad_set, render_campaign, render_creative, render_insight

DOCUMENT_SOURCE_ATTR = "cognee_document_source"


def _date_window(window: str) -> tuple[str, str]:
    today = date.today()
    if window.startswith("last_") and window.endswith("d"):
        days = int(window[5:-1])
        return (today - timedelta(days=days)).isoformat(), today.isoformat()
    raise ValueError("window must use the last_Nd format, for example last_30d")


def _map_resource(resource: Any, mapper) -> Any:
    resource.add_step(mapper)
    resource.apply_hints(primary_key="id", write_disposition="merge")
    return resource


def _stamp_account(account_id: str):
    def stamp(row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["account_id"] = account_id
        return row

    return stamp


def meta_ads_source(
    access_token: str,
    account_ids: list[str] | tuple[str, ...],
    window: str = "last_30d",
    api_version: str = GRAPH_API_VERSION,
):
    """Build Meta Ads resources through dlt's verified Facebook Ads source."""
    if not access_token:
        raise ValueError("access_token is required")
    if not account_ids:
        raise ValueError("at least one Meta ad account ID is required")
    since, until = _date_window(window)
    try:
        from facebook_ads import facebook_ads_source, facebook_insights_source
    except ImportError as exc:
        raise ImportError(
            "Install the connector with its dependencies to use the Meta Ads source"
        ) from exc

    resources = []
    for account_id in account_ids:
        account = str(account_id).removeprefix("act_")
        ads = facebook_ads_source(
            account_id=account,
            access_token=access_token,
            app_api_version=api_version,
        )
        bindings = (
            ("campaigns", CAMPAIGN_FIELDS, render_campaign),
            ("ad_sets", AD_SET_FIELDS, render_ad_set),
            ("ads", AD_FIELDS, render_ad),
            ("ad_creatives", CREATIVE_FIELDS, render_creative),
        )
        for name, fields, mapper in bindings:
            resource = getattr(ads, name)
            resource.bind(fields=fields, states=ACTIVE_AND_ARCHIVED_STATES)
            resource.add_step(_stamp_account(account))
            _map_resource(resource, mapper)
            resource.name = f"{name}_{account}"
            resources.append(resource)

        insights = facebook_insights_source(
            account_id=account,
            access_token=access_token,
            initial_load_past_days=max(1, (date.fromisoformat(until) - date.fromisoformat(since)).days),
            time_increment_days=max(1, (date.fromisoformat(until) - date.fromisoformat(since)).days),
            breakdowns="ads_insights",
            level="ad",
            fields=INSIGHT_FIELDS,
            app_api_version=api_version,
        )
        insights.add_step(_stamp_account(account))
        _map_resource(insights, render_insight)
        insights.name = f"insights_{account}"
        resources.append(insights)

    try:
        import dlt

        combined = dlt.source(name="meta_ads")(lambda: resources)()
        setattr(combined, DOCUMENT_SOURCE_ATTR, "meta_ads")
        setattr(combined, "cognee_pipeline_scope", "meta_ads:" + ",".join(map(str, account_ids)))
        return combined
    except ImportError as exc:
        raise ImportError("Install dlt to use the Meta Ads source") from exc
