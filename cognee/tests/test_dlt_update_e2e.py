"""E2E: update() replaces a DLT source manifest through the full rebuild.

A DLT manifest is built by the DLT cognify route, which writes row nodes and
never a document chunk, so the chunk-level path refuses and update() falls
back to the full rebuild: memory dropped, pinned re-add, cognify. The re-add carries the replacement
resource wrapped in a DataItem pinned to the manifest's id; the resolver must
unwrap it and the DLT route must re-emit the new rows under the same manifest.

Runs on the default local stack; the DLT route makes no LLM calls, but the
rows are embedded, so an embedding configuration (or MOCK_EMBEDDING) is needed.
"""

import asyncio
import json

import dlt

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()

DATASET_NAME = "dlt_update_e2e"
ROWS_V1 = [{"id": 1, "name": "Ada Lovelace"}, {"id": 2, "name": "Alan Turing"}]
ROWS_V2 = [
    {"id": 1, "name": "Ada Lovelace"},
    {"id": 2, "name": "Alan M. Turing"},
    {"id": 3, "name": "Grace Hopper"},
]


def _system_metadata(row) -> dict:
    metadata = getattr(row, "system_metadata", None)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata or {}


async def _row_names(dataset_id, user):
    from cognee.context_global_variables import set_database_global_context_variables

    async with set_database_global_context_variables(dataset_id, user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    return sorted(
        str(props.get("name"))
        for _, props in nodes
        if props.get("type") == "DltRow" or str(props.get("table_name", "")) == "people"
    )


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(dlt.resource(ROWS_V1, name="people", primary_key="id"), DATASET_NAME)
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == DATASET_NAME)
    await cognee.cognify(datasets=[dataset.id])

    (manifest,) = await get_dataset_data(dataset.id)
    assert _system_metadata(manifest).get("source") == "dlt_source", _system_metadata(manifest)
    assert _system_metadata(manifest).get("row_count") == len(ROWS_V1)
    names_v1 = await _row_names(dataset.id, user)
    assert names_v1, "the DLT route emitted no row nodes"

    result = await cognee.update(
        dlt.resource(ROWS_V2, name="people", primary_key="id"),
        dataset.id,
        data_id=manifest.id,
        user=user,
    )
    assert result["status"] == "full_rebuild", result
    assert result["fallback"]["reason"] == "no_baseline", result["fallback"]
    assert "dlt_source cognify route" in result["fallback"]["detail"], result["fallback"]
    assert result["data_id"] == manifest.id and result["regions"] is None, result

    rows = await get_dataset_data(dataset.id)
    assert len(rows) == 1 and rows[0].id == manifest.id, (
        f"the manifest must survive the update under its id: {[r.id for r in rows]}"
    )
    assert _system_metadata(rows[0]).get("row_count") == len(ROWS_V2), _system_metadata(rows[0])

    names_v2 = await _row_names(dataset.id, user)
    assert names_v2 != names_v1, "the graph must reflect the replacement rows"

    # A replacement under another source name is a different manifest: refused
    # before the rebuild deletes anything, so the document survives.
    from cognee.exceptions import CogneeValidationError

    try:
        await cognee.update(
            dlt.resource(ROWS_V2, name="staff", primary_key="id"),
            dataset.id,
            data_id=manifest.id,
            user=user,
        )
    except CogneeValidationError as error:
        assert error.name == "DltSourceIdentityMismatch", error
    else:
        raise AssertionError("a renamed dlt source must be refused")
    survivors = await get_dataset_data(dataset.id)
    assert [row.id for row in survivors] == [manifest.id], (
        "the refusal must not delete the manifest"
    )
    logger.info("DLT update e2e passed: %d -> %d rows", len(ROWS_V1), len(ROWS_V2))


if __name__ == "__main__":
    asyncio.run(main())
