from cognee_community_connector_meta_ads.transform import (
    creative_fingerprint,
    render_ad,
    render_ad_set,
    render_campaign,
    render_insight,
)


def test_campaign_and_ad_set_text_keeps_lineage():
    campaign = render_campaign({
        "id": "c1", "account_id": "a1", "name": "June launch",
        "objective": "OUTCOME_SALES", "status": "PAUSED", "daily_budget": "1000",
    })
    ad_set = render_ad_set({
        "id": "s1", "account_id": "a1", "campaign_id": "c1", "name": "Nairobi music fans",
        "targeting": {
            "age_min": 25, "age_max": 45,
            "geo_locations": {"cities": [{"name": "Nairobi"}]},
            "interests": [{"name": "Live music"}],
            "publisher_platforms": ["facebook", "instagram"],
            "instagram_positions": ["story"],
        },
    })
    assert "June launch" in campaign["content"]
    assert campaign["campaign_id"] == "c1"
    assert "Nairobi" in ad_set["content"]
    assert "instagram" in ad_set["content"]
    assert ad_set["campaign_id"] == "c1"
    assert ad_set["account_id"] == "a1"


def test_ad_and_insight_text_keeps_parent_ids():
    ad = render_ad(
        {"id": "ad1", "account_id": "a1", "campaign_id": "c1", "adset_id": "s1", "name": "Version A"},
        {"id": "cr1", "object_story_spec": {"link_data": {
            "message": "Join us this June", "name": "Meet the community",
            "description": "Tickets available", "link": "https://example.com",
            "call_to_action": {"type": "SIGN_UP"},
        }}},
    )
    insight = render_insight({
        "ad_id": "ad1", "account_id": "a1", "campaign_id": "c1", "adset_id": "s1",
        "date_start": "2025-06-01", "date_stop": "2025-06-30", "spend": "340",
        "clicks": "2100", "ctr": "1.8", "cpc": "0.16",
    })
    assert "Join us this June" in ad["content"]
    assert ad["ad_id"] == "ad1"
    assert "2100" in insight["content"]
    assert insight["campaign_id"] == "c1"
    assert insight["adset_id"] == "s1"
    assert insight["ad_id"] == "ad1"


def test_creative_fingerprint_matches_identical_content():
    first = {"object_story_spec": {"link_data": {"message": "Same copy"}}}
    second = {"object_story_spec": {"link_data": {"message": "Same copy"}}}
    assert creative_fingerprint(first) == creative_fingerprint(second)
