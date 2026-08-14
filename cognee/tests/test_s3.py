import logging

import cognee
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from collections import Counter


logger = get_logger()


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    s3_input = "s3://samples3input"
    await cognee.add(s3_input)

    await cognee.cognify()

    graph_engine = await get_graph_engine()
    graph = await graph_engine.get_graph_data()

    type_counts = Counter(node_data[1].get("type", {}) for node_data in graph[0])

    edge_type_counts = Counter(edge_type[2] for edge_type in graph[1])

    logging.info(type_counts)
    logging.info(edge_type_counts)

    # Assert there is exactly one TextDocument.
    assert type_counts.get("TextDocument", 0) == 2, (
        f"Expected exactly one TextDocument, but found {type_counts.get('TextDocument', 0)}"
    )

    # Assert there are at least two DocumentChunk nodes.
    assert type_counts.get("DocumentChunk", 0) >= 2, (
        f"Expected at least two DocumentChunk nodes, but found {type_counts.get('DocumentChunk', 0)}"
    )

    # Assert there is at least two TextSummary.
    assert type_counts.get("TextSummary", 0) >= 2, (
        f"Expected at least two TextSummary, but found {type_counts.get('TextSummary', 0)}"
    )

    # Assert there is at least one Entity.
    assert type_counts.get("Entity", 0) > 0, (
        f"Expected more than zero Entity nodes, but found {type_counts.get('Entity', 0)}"
    )

    # Assert there is at least one EntityType.
    assert type_counts.get("EntityType", 0) > 0, (
        f"Expected more than zero EntityType nodes, but found {type_counts.get('EntityType', 0)}"
    )

    # Assert that there are at least two 'is_part_of' edges.
    assert edge_type_counts.get("is_part_of", 0) >= 2, (
        f"Expected at least two 'is_part_of' edges, but found {edge_type_counts.get('is_part_of', 0)}"
    )

    # Assert that there are at least two 'made_from' edges.
    assert edge_type_counts.get("made_from", 0) >= 2, (
        f"Expected at least two 'made_from' edges, but found {edge_type_counts.get('made_from', 0)}"
    )

    # Assert that there is at least one 'is_a' edge.
    assert edge_type_counts.get("is_a", 0) >= 1, (
        f"Expected at least one 'is_a' edge, but found {edge_type_counts.get('is_a', 0)}"
    )

    # Assert that there is at least one 'contains' edge.
    assert edge_type_counts.get("contains", 0) >= 1, (
        f"Expected at least one 'contains' edge, but found {edge_type_counts.get('contains', 0)}"
    )

    await dlt_csv_from_s3()


DLT_CSV_BUCKET = "github-runner-cognee-tests"
DLT_CSV_DATASET = "s3_csv_dlt_ds"
DLT_CSV_ROWS = "id,name,section\n1,anemometer,weather\n2,barometer,weather\n3,seismograph,geology\n"


async def dlt_csv_from_s3():
    """A bare s3://...csv path through plain add()/cognify() must take the
    DLT route via the loader engine: data_item_to_text_file localizes the
    object through cognee's storage layer (S3Config credentials), the
    dlt_csv_loader stages it, and one manifest record with a DltRow per line
    lands in the graph.

    The CSV is self-provisioned into a run-scoped key (the bucket is shared
    across CI runs) and removed afterwards. Requires the dlt extra (the job
    installs it) and the job's AWS credentials — the same ones the S3 file
    storage tests write with.
    """
    import uuid

    import dlt  # noqa: F401 — hard requirement; the CI job installs the extra
    import s3fs

    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user
    from cognee.tasks.ingestion.dlt_utils import is_dlt_source_manifest

    fs = s3fs.S3FileSystem()
    key = f"{DLT_CSV_BUCKET}/dlt_csv_ingestion/{uuid.uuid4().hex}/instruments.csv"
    with fs.open(key, "w") as remote_file:
        remote_file.write(DLT_CSV_ROWS)

    try:
        s3_csv_url = f"s3://{key}"
        logging.info("Adding CSV from %s through the DLT loader path", s3_csv_url)
        await cognee.add([s3_csv_url], dataset_name=DLT_CSV_DATASET)
        await cognee.cognify(datasets=[DLT_CSV_DATASET])
    finally:
        fs.rm(key)

    user = await get_default_user()
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DLT_CSV_DATASET]
        )
    )[0]

    manifests = [
        record for record in await get_dataset_data(dataset.id) if is_dlt_source_manifest(record)
    ]
    assert len(manifests) == 1, f"Expected 1 DLT manifest for the S3 CSV, found {len(manifests)}"
    assert manifests[0].system_metadata.get("row_count") == 3, (
        f"Expected row_count 3, got {manifests[0].system_metadata.get('row_count')}"
    )
    assert manifests[0].loader_engine == "dlt_csv_loader", (
        f"Expected the dlt_csv_loader to own the record, got {manifests[0].loader_engine!r}"
    )

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()
        nodes, _ = await graph_engine.get_graph_data()
    csv_census = Counter(props.get("type") for _, props in nodes)
    assert csv_census.get("DltRow", 0) == 3, (
        f"Expected 3 DltRow nodes from the S3 CSV, found {csv_census.get('DltRow', 0)}"
    )
    assert csv_census.get("SchemaTable", 0) == 1, (
        f"Expected 1 SchemaTable node, found {csv_census.get('SchemaTable', 0)}"
    )
    logging.info("S3 CSV DLT ingestion verified: %s", dict(csv_census))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
