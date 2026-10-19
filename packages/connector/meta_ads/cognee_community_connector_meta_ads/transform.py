import hashlib
import json
from typing import Any

from .models import MetaAdsDocument


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _document(row: dict[str, Any], object_type: str, prose: str) -> MetaAdsDocument:
    adset_id = row.get("adset_id") or row.get("adset", {}).get("id")
    ad_id = row.get("ad_id") or (row.get("id") if object_type in {"ad", "insight"} else None)
    campaign_id = row.get("campaign_id") or row.get("campaign", {}).get("id")
    if object_type == "campaign":
        campaign_id = row.get("id")
    lineage = [f"account {row.get('account_id', 'unknown')}"]
    if campaign_id:
        lineage.append(f"campaign {campaign_id}")
    if adset_id:
        lineage.append(f"ad set {adset_id}")
    if ad_id:
        lineage.append(f"ad {ad_id}")
    prose = f"{prose.rstrip('.')} (Lineage: {', '.join(lineage)}.)"
    return {
        "id": str(row["id"]),
        "title": f"Meta Ads {object_type}: {row.get('name') or row['id']}",
        "content": prose,
        "document": prose,
        "object_type": object_type,
        "account_id": str(row.get("account_id", "")),
        "campaign_id": str(campaign_id or ""),
        "adset_id": str(adset_id) if adset_id else None,
        "ad_id": str(ad_id) if ad_id else None,
        "updated_time": row.get("updated_time"),
        "status": row.get("effective_status") or row.get("status"),
        "raw": row,
    }


def render_campaign(row: dict[str, Any]) -> MetaAdsDocument:
    budget = row.get("lifetime_budget") or row.get("daily_budget")
    budget_kind = "lifetime" if row.get("lifetime_budget") else "daily"
    details = [f"Campaign {row.get('name', row['id'])}"]
    if row.get("objective"):
        details.append(f"objective: {_text(row['objective'])}")
    if row.get("effective_status") or row.get("status"):
        details.append(f"status: {_text(row.get('effective_status') or row.get('status'))}")
    if budget:
        details.append(f"{budget_kind} budget: {budget}")
    return _document(row, "campaign", ", ".join(details) + ".")


def _targeting_text(targeting: dict[str, Any]) -> str:
    parts: list[str] = []
    age_min, age_max = targeting.get("age_min"), targeting.get("age_max")
    if age_min or age_max:
        parts.append(f"people aged {age_min or 'unspecified'}–{age_max or 'unspecified'}")
    geo = targeting.get("geo_locations", {})
    places = []
    for key in ("cities", "regions", "countries", "zips", "custom_locations"):
        for place in geo.get(key, []) or []:
            if isinstance(place, dict):
                label = place.get("name") or place.get("key") or place.get("country")
                if label:
                    places.append(str(label))
    if places:
        parts.append("in " + ", ".join(dict.fromkeys(places)))
    interests = [item.get("name") for item in targeting.get("interests", []) if item.get("name")]
    behaviors = [item.get("name") for item in targeting.get("behaviors", []) if item.get("name")]
    if interests or behaviors:
        labels = interests + behaviors
        parts.append("interested in " + ", ".join(dict.fromkeys(labels)))
    platforms = targeting.get("publisher_platforms", [])
    positions = targeting.get("facebook_positions", []) + targeting.get("instagram_positions", [])
    if platforms:
        detail = ", ".join(dict.fromkeys(platforms))
        if positions:
            detail += " " + ", ".join(dict.fromkeys(positions))
        parts.append("shown on " + detail)
    return ", ".join(parts)


def render_ad_set(row: dict[str, Any]) -> MetaAdsDocument:
    targeting = _targeting_text(row.get("targeting") or {})
    details = [f"Ad set {row.get('name', row['id'])}"]
    if row.get("campaign_id"):
        details.append(f"in campaign {row['campaign_id']}")
    if targeting:
        details.append(targeting)
    return _document(row, "ad_set", ", ".join(details) + ".")


def creative_fingerprint(row: dict[str, Any]) -> str:
    story = row.get("object_story_spec") or row.get("asset_feed_spec") or row
    return hashlib.sha256(_text(story).encode("utf-8")).hexdigest()


def render_ad(row: dict[str, Any], creative: dict[str, Any] | None = None) -> MetaAdsDocument:
    creative = creative or row.get("creative") or {}
    story = creative.get("object_story_spec", {})
    link_data = story.get("link_data", {})
    video_data = story.get("video_data", {})
    text = link_data.get("message") or video_data.get("message") or ""
    headline = link_data.get("name") or video_data.get("title") or ""
    description = link_data.get("description") or ""
    cta = link_data.get("call_to_action", {}).get("type") or video_data.get("call_to_action", {}).get("type")
    url = link_data.get("link") or video_data.get("call_to_action", {}).get("value", {}).get("link")
    fields = [f"Ad {row.get('name', row['id'])}"]
    if text:
        fields.append(f"primary text: {_text(text)}")
    if headline:
        fields.append(f"headline: {_text(headline)}")
    if description:
        fields.append(f"description: {_text(description)}")
    if cta:
        fields.append(f"call to action: {_text(cta)}")
    if url:
        fields.append(f"destination: {_text(url)}")
    result = _document(row, "ad", ". ".join(fields) + ".")
    if creative.get("id"):
        result["raw"] = {**row, "creative_id": creative["id"]}
    return result


def render_creative(row: dict[str, Any]) -> MetaAdsDocument:
    story = row.get("object_story_spec") or {}
    link_data = story.get("link_data") or {}
    video_data = story.get("video_data") or {}
    parts = [f"Ad creative {row.get('name') or row['id']}"]
    for key, label in (("message", "primary text"), ("name", "headline"), ("description", "description")):
        value = link_data.get(key)
        if value:
            parts.append(f"{label}: {_text(value)}")
    if video_data.get("message"):
        parts.append(f"primary text: {_text(video_data['message'])}")
    if link_data.get("link"):
        parts.append(f"destination: {_text(link_data['link'])}")
    return _document(row, "creative", ". ".join(parts) + ".")


def render_insight(row: dict[str, Any]) -> MetaAdsDocument:
    window = row.get("date_start", "")
    if row.get("date_stop"):
        window += f" to {row['date_stop']}"
    parts = [f"Performance for ad {row.get('ad_id', row.get('id', 'unknown'))} over {window}".strip()]
    for key, label in (("spend", "spent"), ("clicks", "got clicks"), ("impressions", "received impressions"), ("ctr", "CTR"), ("cpc", "CPC")):
        if row.get(key) not in (None, ""):
            parts.append(f"{label} {row[key]}")
    row_with_id = dict(row)
    row_with_id.setdefault("id", f"{row.get('ad_id', '')}:{row.get('date_start', '')}:{row.get('date_stop', '')}")
    return _document(row_with_id, "insight", ", ".join(parts) + ".")
