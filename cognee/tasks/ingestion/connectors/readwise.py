"""Readwise connector for cognee, a ``dlt`` source that turns highlights into memory.

Sync a set of explicit Readwise resources into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import readwise_source

    await cognee.remember(
        readwise_source(),
        dataset_name="my_readwise_data",
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
