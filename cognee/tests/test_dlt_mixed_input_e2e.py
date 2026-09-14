"""E2E test: mixed DLT source + regular text input in one dataset.

Verifies the DLT single-item pipeline:
- a DLT resource resolves to a single manifest Data record with rows
  deduplicated by (table, pk, content_hash)
- a file-like CSV upload (the API's UploadFile shape) in the same add()
  resolves to its own manifest, named after the uploaded filename
- a regular text item in the same add() keeps its own Data record
- cognify executes all of them under ONE cognify_pipeline run, resolving the
  task list per item: the manifests through the deterministic DLT task list
  (row chunks + schema graph, no LLM) and the text through the standard list
  (LLM entity extraction and summarization)
- both are searchable afterwards
"""

import asyncio
import io
import json

import dlt
from starlette.datastructures import UploadFile

import cognee
from cognee import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger()

DATASET_NAME = "dlt_mixed_e2e"

TEXT_FACT = (
    "Zorblatt Industries manufactures underwater bicycles in Reykjavik. "
    "The company was founded by Melinda Voss in 1997."
)

# id=3 appears twice with identical content — the manifest must collapse it.
PEOPLE_ROWS = [
    {"id": 1, "name": "Ada Lovelace", "specialty": "analytical engines"},
    {"id": 2, "name": "Alan Turing", "specialty": "computability"},
    {"id": 3, "name": "Grace Hopper", "specialty": "compilers"},
    {"id": 3, "name": "Grace Hopper", "specialty": "compilers"},
]
UNIQUE_ROW_COUNT = 3

# Uploaded as a file-like CSV (the shape the API's /add route receives) —
# must resolve to its own manifest named after the filename stem.
MACHINES_CSV = b"id,hostname,room\n1,turing-01,lab-a\n2,hopper-02,lab-b\n"
MACHINES_ROW_COUNT = 2
MACHINES_SOURCE_NAME = "lab_machines"


def _system_metadata(row) -> dict:
    ext = row["system_metadata"]
    if isinstance(ext, str):
        ext = json.loads(ext)
    return ext or {}


async def test_dlt_mixed_input():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    people_resource = dlt.resource(PEOPLE_ROWS, name="people", primary_key="id")
    machines_upload = UploadFile(io.BytesIO(MACHINES_CSV), filename="Lab Machines.csv")

    await cognee.add([people_resource, TEXT_FACT, machines_upload], DATASET_NAME)

    # --- Add: two manifest Data records + one text Data record --------------
    relational_engine = get_relational_engine()
    data_rows = await relational_engine.get_all_data_from_table("data")
    assert len(data_rows) == 3, (
        f"Expected 3 data records (2 DLT manifests + 1 text), got {len(data_rows)}"
    )

    manifest_rows = [r for r in data_rows if _system_metadata(r).get("source") == "dlt_source"]
    assert len(manifest_rows) == 2, (
        f"Expected exactly 2 DLT-source manifest records, got {len(manifest_rows)}"
    )
    manifests_by_source = {
        _system_metadata(r)["source_name"]: _system_metadata(r) for r in manifest_rows
    }

    manifest_meta = manifests_by_source["people"]
    assert manifest_meta["row_count"] == UNIQUE_ROW_COUNT, (
        f"Duplicate row was not collapsed: row_count={manifest_meta['row_count']}, "
        f"expected {UNIQUE_ROW_COUNT}"
    )
    assert "people" in manifest_meta["tables"], (
        f"Manifest tables missing 'people': {manifest_meta['tables']}"
    )

    # The uploaded CSV gets its own manifest, named after the filename stem.
    assert MACHINES_SOURCE_NAME in manifests_by_source, (
        f"Uploaded CSV manifest missing: sources={sorted(manifests_by_source)}"
    )
    machines_meta = manifests_by_source[MACHINES_SOURCE_NAME]
    assert machines_meta["row_count"] == MACHINES_ROW_COUNT, (
        f"Uploaded CSV row_count={machines_meta['row_count']}, expected {MACHINES_ROW_COUNT}"
    )

    # --- Cognify: manifest → DLT pipeline, text → standard pipeline ---------
    await cognee.cognify(datasets=[DATASET_NAME])

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    node_props = [props for _, props in nodes]
    schema_tables = [p for p in node_props if p.get("type") == "SchemaTable"]
    assert any(p.get("name") == "people" for p in schema_tables), (
        "SchemaTable node for 'people' missing — DLT pipeline did not build the schema graph"
    )
    assert any(p.get("name") == MACHINES_SOURCE_NAME for p in schema_tables), (
        f"SchemaTable node for '{MACHINES_SOURCE_NAME}' missing — uploaded CSV did not "
        "take the DLT pipeline"
    )

    expected_row_edges = UNIQUE_ROW_COUNT + MACHINES_ROW_COUNT
    is_row_of_count = sum(1 for edge in edges if edge[2] == "is_row_of")
    assert is_row_of_count == expected_row_edges, (
        f"Expected {expected_row_edges} is_row_of edges (one per unique row), got {is_row_of_count}"
    )

    summaries = [p for p in node_props if p.get("type") == "TextSummary"]
    assert summaries, (
        "No TextSummary nodes — the regular text item did not run the LLM cognify pipeline"
    )

    entities = [p for p in node_props if p.get("type") == "Entity"]
    assert entities, "No Entity nodes — LLM extraction did not run for the regular text item"

    # --- Search: chunk search is documents-only; rows have their own
    # collection (DltRow_text) read by row-aware retrieval. ----------
    chunk_result = await cognee.search(
        "Ada Lovelace", query_type=SearchType.CHUNKS, datasets=[DATASET_NAME]
    )
    assert "analytical engines" not in str(chunk_result), (
        f"DLT rows must NOT surface in chunk search (documents only): {str(chunk_result)[:500]}"
    )

    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET_NAME]
        )
    )[0]
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        vector_engine = await get_vector_engine_async()
        row_hits = await vector_engine.search(
            "DltRow_text", query_text="Ada Lovelace", limit=3, include_payload=True
        )
    assert any("analytical engines" in str(hit.payload) for hit in row_hits), (
        "DLT row text must be searchable in its own DltRow_text collection"
    )

    text_result = await cognee.search(
        "underwater bicycles", query_type=SearchType.CHUNKS, datasets=[DATASET_NAME]
    )
    assert "Zorblatt" in str(text_result), (
        f"Text content not retrievable via chunk search: {str(text_result)[:500]}"
    )

    logger.info("DLT mixed input e2e test passed.")


if __name__ == "__main__":
    asyncio.run(test_dlt_mixed_input())
