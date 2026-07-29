"""SaaS data-source connectors for cognee.

Connectors that pull an external source (Gmail, Slack, Notion, Google Drive,
Confluence, …) into cognee memory are distributed as **separate community
packages** under https://github.com/topoteretes/cognee-community
(``cognee-community-connector-<source>``), so core stays free of per-source SDKs.

Each connector is a plain ``dlt`` source you hand to ``cognee.remember(...)``. It
reuses core's DLT ingestion path (``resolve_dlt_sources`` -> ``ingest_dlt_source``
-> ``orphan_cleanup``) for incremental re-sync and forget-on-source-deletion, and
opts into the document ingestion path via ``dlt_utils.DOCUMENT_SOURCE_ATTR``. No
connector is bundled in core::

    pip install cognee-community-connector-gmail
    from cognee_community_connector_gmail import gmail_source

    await cognee.remember(gmail_source(...), dataset_name="gmail_inbox")
"""

__all__: list[str] = []
