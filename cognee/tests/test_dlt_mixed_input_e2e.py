"""E2E test: mixed DLT source + regular text input in one dataset.

Verifies the DLT single-item pipeline:
- a DLT resource resolves to a single manifest Data record with rows
  deduplicated by (table, pk, content_hash)
- a regular text item in the same add() keeps its own Data record
- cognify routes the manifest to dlt_cognify_pipeline (deterministic row
  chunks + schema graph, no LLM) and the text to cognify_pipeline (LLM
  entity extraction and summarization)
- both are searchable afterwards
"""

import asyncio
import json

import dlt

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


def _external_metadata(row) -> dict:
    ext = row["external_metadata"]
    if isinstance(ext, str):
        ext = json.loads(ext)
    return ext or {}


async def test_dlt_mixed_input():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    people_resource = dlt.resource(PEOPLE_ROWS, name="people", primary_key="id")

    await cognee.add([people_resource, TEXT_FACT], DATASET_NAME)

    # --- Add: one manifest Data record + one text Data record ---------------
    relational_engine = get_relational_engine()
    data_rows = await relational_engine.get_all_data_from_table("data")
    assert len(data_rows) == 2, (
        f"Expected 2 data records (1 DLT manifest + 1 text), got {len(data_rows)}"
    )

    manifest_rows = [r for r in data_rows if _external_metadata(r).get("source") == "dlt_source"]
    assert len(manifest_rows) == 1, (
        f"Expected exactly 1 DLT-source manifest record, got {len(manifest_rows)}"
    )

    manifest_meta = _external_metadata(manifest_rows[0])
    assert manifest_meta["row_count"] == UNIQUE_ROW_COUNT, (
        f"Duplicate row was not collapsed: row_count={manifest_meta['row_count']}, "
        f"expected {UNIQUE_ROW_COUNT}"
    )
    assert "people" in manifest_meta["tables"], (
        f"Manifest tables missing 'people': {manifest_meta['tables']}"
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

    is_row_of_count = sum(1 for edge in edges if edge[2] == "is_row_of")
    assert is_row_of_count == UNIQUE_ROW_COUNT, (
        f"Expected {UNIQUE_ROW_COUNT} is_row_of edges (one per unique row), "
        f"got {is_row_of_count}"
    )

    summaries = [p for p in node_props if p.get("type") == "TextSummary"]
    assert summaries, (
        "No TextSummary nodes — the regular text item did not run the LLM cognify pipeline"
    )

    entities = [p for p in node_props if p.get("type") == "Entity"]
    assert entities, (
        "No Entity nodes — LLM extraction did not run for the regular text item"
    )

    # --- Search: both the DLT rows and the text are retrievable -------------
    row_result = await cognee.search(
        "Ada Lovelace", query_type=SearchType.CHUNKS, datasets=[DATASET_NAME]
    )
    assert "analytical engines" in str(row_result), (
        f"DLT row not retrievable via chunk search: {str(row_result)[:500]}"
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
