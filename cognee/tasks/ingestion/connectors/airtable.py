"""Airtable connector for cognee, a ``dlt`` source that turns Airtable records into memory.

Sync a set of explicit Airtable bases/tables into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import airtable_source

    await cognee.remember(
        airtable_source(
            base_ids=["appXXXXXXX"],
        ),
        dataset_name="my_airtable_data",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
"""

import logging
from collections.abc import Iterator

import httpx

from cognee.tasks.ingestion import dlt_utils

logger = logging.getLogger(__name__)

AIRTABLE_API_URL = "https://api.airtable.com/v0"


def _get_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def airtable_source(
    token: str | None = None,
    base_ids: list[str] | None = None,
    resource_name: str = "airtable_records",
):
    """Create a dlt source yielding Airtable records.

    Args:
        token: Airtable Personal Access Token (or OAuth token). Falls back to AIRTABLE_API_KEY.
        base_ids: List of base IDs to sync. If empty, syncs all bases available to the token.
        resource_name: The dlt resource / table name to yield rows into.
    """
    import os

    try:
        import dlt
    except ImportError as e:
        raise ImportError("The Airtable connector requires dlt: pip install dlt") from e

    token = token or os.environ.get("AIRTABLE_API_KEY")
    if not token:
        raise ValueError(
            "Airtable token must be provided or set in AIRTABLE_API_KEY environment variable"
        )

    @dlt.resource(name=resource_name, write_disposition="merge", primary_key="id")
    def airtable_records(
        last_updated_time: dlt.sources.incremental = dlt.sources.incremental("last_edited_time"),  # noqa: B008
    ) -> Iterator[dict]:
        headers = _get_headers(token)
        bases = base_ids

        # If no bases provided, fetch all bases (requires meta/bases scope)
        if not bases:
            response = httpx.get(f"{AIRTABLE_API_URL}/meta/bases", headers=headers, timeout=30)
            response.raise_for_status()
            bases = [b["id"] for b in response.json().get("bases", [])]

        for base_id in bases:
            # Fetch tables for the base
            tables_response = httpx.get(
                f"{AIRTABLE_API_URL}/meta/bases/{base_id}/tables", headers=headers, timeout=30
            )
            tables_response.raise_for_status()
            tables = tables_response.json().get("tables", [])

            for table in tables:
                table_id = table["id"]
                offset = None

                while True:
                    url = f"{AIRTABLE_API_URL}/{base_id}/{table_id}"
                    params = {}
                    if offset:
                        params["offset"] = offset

                    # Incremental filter could go here if Airtable supported a global updated filter easily,
                    # but typically we fetch all or rely on formula. For simplicity, we yield and let dlt filter.

                    resp = httpx.get(url, headers=headers, params=params, timeout=30)
                    if resp.status_code == 404:
                        break  # Table might have been deleted or token lacks access
                    resp.raise_for_status()

                    data = resp.json()
                    records = data.get("records", [])

                    for record in records:
                        # Extract the content
                        row_id = record.get("id")
                        created_time = record.get("createdTime")
                        fields = record.get("fields", {})

                        # Convert fields to a Markdown string
                        content_lines = [f"Base: {base_id}, Table: {table['name']}"]
                        for k, v in fields.items():
                            content_lines.append(f"{k}: {v}")

                        yield {
                            "id": f"{base_id}_{table_id}_{row_id}",
                            "url": f"https://airtable.com/{base_id}/{table_id}/{row_id}",
                            "title": f"Airtable Record {row_id}",
                            "content": "\n".join(content_lines),
                            "last_edited_time": created_time,  # Simplify by using createdTime if lastModified not available
                            "base_id": base_id,
                            "table_id": table_id,
                            "record_id": row_id,
                        }

                    offset = data.get("offset")
                    if not offset:
                        break

    @dlt.source(name="airtable")
    def _airtable():
        return airtable_records

    source = _airtable()
    setattr(source, dlt_utils.DOCUMENT_SOURCE_ATTR, "airtable")
    return source
