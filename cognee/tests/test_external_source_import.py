"""E2E: import a NON-cognee memory export (Zep/Graphiti graph shape) and
search it.

The cross-provider half of the COGX contract, driven through the public
``cognee.remember(ZepSource(...))`` path against real stores: external
entities get cognee's CURRENT class-namespaced ids (``Entity.id_for(name)``,
never the source system's ids), same-named records merge into one node,
facts that resolve onto one edge key are deduplicated (reported in the
import summary), labels become EntityType nodes, and the imported graph
answers a search.

Runs single-tenant: external imports are identity-agnostic (name-keyed), and
the multi-user posture is covered by test_cogx_roundtrip.py.
"""

import asyncio
import json
import os
import pathlib
import tempfile
from uuid import uuid4

import cognee
from cognee.modules.migration.sources.zep import ZepSource
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()

QUERY = "Who did Ada Lovelace collaborate with?"


def build_zep_export(path: pathlib.Path) -> None:
    """A realistic Zep/Graphiti graph export: UUID-keyed entities, edges by
    node uuid, one episode. Two same-named entities and two facts that
    resolve onto one edge key exercise merging + dedup."""
    ada_one, ada_two, babbage = str(uuid4()), str(uuid4()), str(uuid4())
    export = {
        "episodes": [
            {
                "uuid": str(uuid4()),
                "name": "history-chat",
                "content": "We discussed how Ada Lovelace collaborated with Charles Babbage.",
                "created_at": "2026-01-01T10:00:00Z",
            }
        ],
        "entities": [
            {"uuid": ada_one, "name": "Ada Lovelace", "labels": ["Person"]},
            {"uuid": ada_two, "name": "Ada Lovelace", "labels": ["Person"]},
            {"uuid": babbage, "name": "Charles Babbage", "labels": ["Person"]},
        ],
        "edges": [
            {
                "uuid": str(uuid4()),
                "source_node_uuid": ada_one,
                "target_node_uuid": babbage,
                "name": "COLLABORATED_WITH",
                "fact": "Ada Lovelace collaborated with Charles Babbage.",
            },
            {
                "uuid": str(uuid4()),
                "source_node_uuid": ada_two,
                "target_node_uuid": babbage,
                "name": "COLLABORATED_WITH",
                "fact": "Lovelace worked with Babbage on the Analytical Engine.",
            },
        ],
    }
    path.write_text(json.dumps(export), encoding="utf-8")


async def main():
    os.environ["AUTO_FEEDBACK"] = "False"
    # Single-tenant: read dynamically, so this wins over any ambient .env.
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    base = pathlib.Path(__file__).parent
    cognee.config.data_root_directory(
        str((base / ".data_storage/test_external_source_import").resolve())
    )
    cognee.config.system_root_directory(
        str((base / ".cognee_system/test_external_source_import").resolve())
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    with tempfile.TemporaryDirectory() as temporary_directory:
        export_path = pathlib.Path(temporary_directory) / "zep_export.json"
        build_zep_export(export_path)

        source = ZepSource(str(export_path), mode="preserve")
        result = await cognee.remember(source, dataset_name="zep_import")
        assert result.status == "completed", f"Import did not complete: {result.status!r}"

        (summary,) = [item for item in result.items if item.get("kind") == "migration_import"]
        logger.info("Import summary: %s", summary)
        assert summary["record_counts"] == {"episode": 1, "entity": 3, "fact": 2}
        assert summary["deduped_edges"] == 1, (
            "Two facts resolving onto one edge key must dedupe to a single edge."
        )
        assert summary["skipped_facts"] == 0

    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.engine.models import Entity

    engine = await get_graph_engine()
    nodes, edges = await engine.get_graph_data()

    adas = [(node_id, props) for node_id, props in nodes if props.get("name") == "Ada Lovelace"]
    assert len(adas) == 1, "Same-named external entities must merge into one node."
    assert str(adas[0][0]) == str(Entity.id_for("Ada Lovelace")), (
        "External entities must use cognee's current class-namespaced ids."
    )
    assert any(
        props.get("type") == "EntityType" and props.get("name") == "Person" for _, props in nodes
    ), "Zep labels must become EntityType nodes."
    collaborated = [edge for edge in edges if edge[2] == "COLLABORATED_WITH"]
    assert len(collaborated) == 1, (
        f"Expected exactly 1 COLLABORATED_WITH edge after dedup, got {len(collaborated)}."
    )

    results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=QUERY, datasets=["zep_import"]
    )
    assert len(results) != 0, "Search on the imported external graph returned nothing."
    logger.info("Search results: %s", results)

    print("External source import e2e passed: Zep export -> import -> search.")


if __name__ == "__main__":
    asyncio.run(main())
