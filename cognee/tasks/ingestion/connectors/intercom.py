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
