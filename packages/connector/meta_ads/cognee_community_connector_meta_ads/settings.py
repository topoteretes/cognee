GRAPH_API_VERSION = "v23.0"

CAMPAIGN_FIELDS = (
    "id", "account_id", "name", "objective", "status", "effective_status",
    "daily_budget", "lifetime_budget", "updated_time",
)
AD_SET_FIELDS = (
    "id", "account_id", "campaign_id", "name", "status", "effective_status",
    "daily_budget", "lifetime_budget", "targeting", "promoted_object", "updated_time",
)
AD_FIELDS = (
    "id", "account_id", "campaign_id", "adset_id", "name", "status",
    "effective_status", "creative", "updated_time",
)
CREATIVE_FIELDS = (
    "id", "account_id", "name", "object_story_spec", "asset_feed_spec",
    "url_tags", "effective_object_story_id", "updated_time",
)
INSIGHT_FIELDS = (
    "account_id", "campaign_id", "adset_id", "ad_id", "date_start", "date_stop",
    "spend", "impressions", "clicks", "ctr", "cpc", "actions",
)
ACTIVE_AND_ARCHIVED_STATES = (
    "ACTIVE", "PAUSED", "PENDING_REVIEW", "DISAPPROVED", "PREAPPROVED",
    "PENDING_BILLING_INFO", "CAMPAIGN_PAUSED", "ADSET_PAUSED", "ARCHIVED", "DELETED",
)
