"""Runnable integration demo for the ``consolidate_entities`` memify pipeline.

It seeds two near-duplicate entities — "New York City" and "NYC" — into the
default local backends (Kuzu graph + LanceDB vectors) using the offline
``fastembed`` embedder (no API key needed), then runs
``consolidate_entities_pipeline`` and verifies the acceptance criteria:

* the two entities collapse into a single canonical node, and
* every original edge survives on the canonical (the canonical-side
  ``located_in -> United States`` and the duplicate-side
  ``contains -> Central Park``), and
* the duplicate's name embedding is purged from the ``Entity_name`` collection.

Run it with the offline embedder::

    cd cognee
    EMBEDDING_PROVIDER=fastembed \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    EMBEDDING_DIMENSIONS=384 \
    uv run python examples/guides/consolidate_entities_example.py

The script exits non-zero if any assertion fails.
"""

import asyncio
import os
import sys

# Default to the offline fastembed embedder so the demo needs no API key.
os.environ.setdefault("EMBEDDING_PROVIDER", "fastembed")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "384")
# This pipeline never calls the LLM, so skip the pipeline's LLM connection
# preflight (otherwise it requires an LLM API key even though it is unused).
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

import cognee  # noqa: E402
from cognee.context_global_variables import (  # noqa: E402
    set_database_global_context_variables,
)
from cognee.infrastructure.databases.graph import get_graph_engine  # noqa: E402
from cognee.infrastructure.databases.vector import get_vector_engine  # noqa: E402
from cognee.memify_pipelines.consolidate_entities import (  # noqa: E402
    consolidate_entities_pipeline,
)
from cognee.modules.engine.models.Entity import Entity  # noqa: E402
from cognee.modules.engine.models.EntityType import EntityType  # noqa: E402
from cognee.modules.engine.operations.setup import setup  # noqa: E402
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (  # noqa: E402
    resolve_authorized_user_datasets,
)
from cognee.tasks.storage.index_data_points import index_data_points  # noqa: E402

DATASET = "consolidate_entities_demo"
DUP_NAMES = {"New York City", "NYC"}


async def _seed():
    """Create the two near-duplicate entities plus distinct neighbors/edges."""
    graph = await get_graph_engine()

    city = EntityType(name="City", description="A populated urban place.")
    country = EntityType(name="Country", description="A sovereign nation.")
    park = EntityType(name="Park", description="A public green space.")

    nycity = Entity(
        name="New York City",
        is_a=city,
        description="The most populous city in the United States.",
    )
    nyc = Entity(name="NYC", is_a=city, description="Common abbreviation for New York City.")
    usa = Entity(name="United States", is_a=country, description="A country in North America.")
    central_park = Entity(
        name="Central Park", is_a=park, description="A large urban park in Manhattan."
    )

    await graph.add_nodes([city, country, park, nycity, nyc, usa, central_park])
    await graph.add_edges(
        [
            (str(nycity.id), str(city.id), "is_a", {}),
            (str(nyc.id), str(city.id), "is_a", {}),
            (str(usa.id), str(country.id), "is_a", {}),
            (str(central_park.id), str(park.id), "is_a", {}),
            # canonical-side edge
            (str(nycity.id), str(usa.id), "located_in", {}),
            # duplicate-side edge — must survive on the canonical after merge
            (str(nyc.id), str(central_park.id), "contains", {}),
        ]
    )
    # Index entity names so detection (cosine) and embedding purge are exercised.
    await index_data_points([city, country, park, nycity, nyc, usa, central_park])
    return {str(nycity.id), str(nyc.id)}


async def _entities(graph):
    nodes, edges = await graph.get_graph_data()
    entities = {str(node_id): props for node_id, props in nodes if props.get("type") == "Entity"}
    return entities, edges


def _check(label, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    return condition


async def main():
    print("== consolidate_entities integration demo (Kuzu + fastembed) ==")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    # Recreate the relational/graph/vector schema (and default user) that the
    # prune above dropped, before resolving datasets.
    await setup()

    user, datasets = await resolve_authorized_user_datasets(DATASET, None)
    target = datasets[0]

    async with set_database_global_context_variables(target.id, target.owner_id):
        dup_ids = await _seed()
        graph = await get_graph_engine()
        entities_before, edges_before = await _entities(graph)
        dup_names_before = sorted(
            props.get("name")
            for props in entities_before.values()
            if props.get("name") in DUP_NAMES
        )
        print(f"Seeded entities: {sorted(p.get('name') for p in entities_before.values())}")
        print(f"Duplicate pair present before: {dup_names_before}")

    # Run the real pipeline (no dry_run).
    await consolidate_entities_pipeline(similarity_threshold=0.6, dataset=DATASET, user=user)

    async with set_database_global_context_variables(target.id, target.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        entities_after, edges_after = await _entities(graph)

        surviving = [
            (eid, props) for eid, props in entities_after.items() if props.get("name") in DUP_NAMES
        ]
        survivor_id = surviving[0][0] if surviving else None
        survivor_name = surviving[0][1].get("name") if surviving else None

        # Edges incident to the survivor, by relationship -> neighbor name.
        neighbor_name = {eid: p.get("name") for eid, p in entities_after.items()}
        # include EntityType names too
        for node_id, props in (await graph.get_graph_data())[0]:
            neighbor_name.setdefault(str(node_id), props.get("name"))
        survivor_rels = set()
        for source, tgt, rel, _ in edges_after:
            if str(source) == survivor_id:
                survivor_rels.add((rel, neighbor_name.get(str(tgt))))

        # Verify the deleted duplicate's embedding is actually gone from
        # Entity_name: embed its name and confirm its id is not among the
        # nearest matches (a real query_vector search, not a no-op probe).
        deleted_ids = dup_ids - {survivor_id}
        purged = True
        for deleted_id in deleted_ids:
            deleted_name = (entities_before.get(deleted_id) or {}).get("name") or survivor_name
            probe_vector = (await vector.embed_data([deleted_name]))[0]
            results = await vector.search(
                "Entity_name", query_text=None, query_vector=probe_vector, limit=50
            )
            present_ids = {str(getattr(r, "id", None)) for r in results}
            if deleted_id in present_ids:
                purged = False

        print(f"Surviving duplicate-pair node: {survivor_name} ({survivor_id})")
        print(f"Survivor outgoing relationships: {sorted(survivor_rels)}")

        ok = True
        ok &= _check("exactly one of {New York City, NYC} remains", len(surviving) == 1)
        ok &= _check(
            "canonical kept its located_in -> United States edge",
            ("located_in", "United States") in survivor_rels,
        )
        ok &= _check(
            "duplicate's contains -> Central Park edge moved to the canonical",
            ("contains", "Central Park") in survivor_rels,
        )
        ok &= _check(
            "United States entity still present",
            any(p.get("name") == "United States" for p in entities_after.values()),
        )
        ok &= _check(
            "Central Park entity still present",
            any(p.get("name") == "Central Park" for p in entities_after.values()),
        )
        ok &= _check("deleted duplicate embedding purged from Entity_name", purged)

    print("== RESULT:", "ALL CHECKS PASSED ==" if ok else "FAILURES PRESENT ==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
