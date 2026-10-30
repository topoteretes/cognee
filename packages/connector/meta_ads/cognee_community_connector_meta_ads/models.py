from typing import Any, TypedDict


class MetaAdsDocument(TypedDict):
    id: str
    document: str
    object_type: str
    account_id: str
    campaign_id: str
    adset_id: str | None
    ad_id: str | None
    updated_time: str | None
    status: str | None
    raw: dict[str, Any]
