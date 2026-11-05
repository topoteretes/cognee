"""Klaviyo connector for cognee, a ``dlt`` source that turns Klaviyo profiles and campaigns into memory.

Sync a set of explicit Klaviyo resources into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import klaviyo_source

    await cognee.remember(
        klaviyo_source(),
        dataset_name="my_klaviyo_data",
        primary_key="id",
        write_disposition="merge",
    )
"""

import logging
import os
from collections.abc import Iterator

import httpx

from cognee.tasks.ingestion import dlt_utils

logger = logging.getLogger(__name__)

KLAVIYO_API_URL = "https://a.klaviyo.com/api"
KLAVIYO_REVISION = "2024-02-15"


def _get_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Klaviyo-API-Key {api_key}",
        "revision": KLAVIYO_REVISION,
        "accept": "application/json",
    }


def klaviyo_source(api_key: str | None = None, resource_name: str = "klaviyo_profiles"):
    """Create a dlt source yielding Klaviyo profiles.

    Args:
        api_key: Klaviyo Private API Key. Falls back to KLAVIYO_API_KEY.
        resource_name: The dlt resource / table name to yield rows into.
    """
    try:
        import dlt
    except ImportError as e:
        raise ImportError("The Klaviyo connector requires dlt: pip install dlt") from e

    api_key = api_key or os.environ.get("KLAVIYO_API_KEY")
    if not api_key:
        raise ValueError(
            "Klaviyo API key must be provided or set in KLAVIYO_API_KEY environment variable"
        )

    @dlt.resource(name=resource_name, write_disposition="merge", primary_key="id")
    def klaviyo_profiles(
        last_updated_time: dlt.sources.incremental = dlt.sources.incremental("updated"),  # noqa: B008
    ) -> Iterator[dict]:
        headers = _get_headers(api_key)
        url = f"{KLAVIYO_API_URL}/profiles"

        while url:
            resp = httpx.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            profiles = data.get("data", [])

            for profile in profiles:
                profile_id = profile.get("id")
                attributes = profile.get("attributes", {})

                content_lines = [f"Klaviyo Profile: {profile_id}"]
                for k, v in attributes.items():
                    if v:
                        content_lines.append(f"{k}: {v}")

                yield {
                    "id": profile_id,
                    "url": f"https://www.klaviyo.com/profile/{profile_id}",
                    "title": f"Klaviyo Profile {attributes.get('email', profile_id)}",
                    "content": "\n".join(content_lines),
                    "updated": attributes.get("updated"),
                }

            # Pagination
            links = data.get("links", {})
            url = links.get("next")

    @dlt.source(name="klaviyo")
    def _klaviyo():
        return klaviyo_profiles

    source = _klaviyo()
    setattr(source, dlt_utils.DOCUMENT_SOURCE_ATTR, "klaviyo")
    return source
