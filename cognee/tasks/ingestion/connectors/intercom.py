"""Intercom connector for cognee, a ``dlt`` source that turns contacts into memory.

Sync a set of explicit Intercom resources into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import intercom_source

    await cognee.remember(
        intercom_source(),
        dataset_name="my_intercom_data",
        primary_key="id",
        write_disposition="merge",
    )
"""

import os
import logging
from typing import Iterator

import httpx

from cognee.tasks.ingestion import dlt_utils

logger = logging.getLogger(__name__)

INTERCOM_API_URL = "https://api.intercom.io"
INTERCOM_VERSION = "2.11"

def _get_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Intercom-Version": INTERCOM_VERSION,
    }
