"""Real-backend integration test for the consolidate_entities pipeline.

Unlike the mocked unit tests in
``cognee/tests/unit/memify_pipelines/test_consolidate_entities_pipeline.py``,
this exercises the whole pipeline against the real default stack (Kuzu graph +
LanceDB vectors) with real embeddings produced by the configured embedding
engine. It seeds two near-duplicate entities ("New York City" / "NYC") with
distinct edges and asserts they collapse into one node, every edge is preserved
on the survivor, and the duplicate's name embedding is removed from the
``Entity_name`` collection.

Runs in the secret-gated integration suite (CI provides the embedding backend).
Locally it can be run against the offline fastembed embedder, e.g.::

    EMBEDDING_PROVIDER=fastembed EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    EMBEDDING_DIMENSIONS=384 COGNEE_SKIP_CONNECTION_TEST=true \
    uv run pytest cognee/tests/integration/tasks/test_consolidate_entities_integration.py -v
"""

import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.low_level import setup as cognee_setup
from cognee.memify_pipelines.consolidate_entities import consolidate_entities_pipeline
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.tasks.storage.index_data_points import index_data_points

DATASET = "consolidate_entities_integration"
DUPLICATE_NAMES = {"New York City", "NYC"}


@pytest_asyncio.fixture
async def clean_test_environment():
    """Isolate this test in its own system/data directories and reset state."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    cognee.config.system_root_directory(
        str(base_dir / ".cognee_system/test_consolidate_entities_integration")
    )
    cognee.config.data_root_directory(
        str(base_dir / ".data_storage/test_consolidate_entities_integration")
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


async def _seed(graph):
    """Seed two near-duplicate cities plus distinct neighbors and edges."""
    city = EntityType(name="City", description="A populated urban place.")
    country = EntityType(name="Country", description="A sovereign nation.")
    park_type = EntityType(name="Park", description="A public green space.")

    nycity = Entity(name="New York City", is_a=city, description="The most populous US city.")
    nyc = Entity(name="NYC", is_a=city, description="Common abbreviation for New York City.")
    usa = Entity(name="United States", is_a=country, description="A country in North America.")
    central_park = Entity(
        name="Central Park", is_a=park_type, description="A large urban park in Manhattan."
    )

    await graph.add_nodes([city, country, park_type, nycity, nyc, usa, central_park])
    await graph.add_edges(
        [
            (str(nycity.id), str(city.id), "is_a", {}),
            (str(nyc.id), str(city.id), "is_a", {}),
            (str(usa.id), str(country.id), "is_a", {}),
            (str(central_park.id), str(park_type.id), "is_a", {}),
            (str(nycity.id), str(usa.id), "located_in", {}),  # canonical-side edge
            (str(nyc.id), str(central_park.id), "contains", {}),  # duplicate-side edge
        ]
    )
    await index_data_points([city, country, park_type, nycity, nyc, usa, central_park])
    return {str(nycity.id), str(nyc.id)}


@pytest.mark.asyncio
async def test_consolidate_entities_merges_real_graph(clean_test_environment):
    user, datasets = await resolve_authorized_user_datasets(DATASET, None)
    target = datasets[0]

    async with set_database_global_context_variables(target.id, target.owner_id):
        graph = await get_graph_engine()
        duplicate_ids = await _seed(graph)
        nodes_before, _ = await graph.get_graph_data()
        name_by_id_before = {str(node_id): props.get("name") for node_id, props in nodes_before}
        # Both duplicates exist before consolidation.
        assert {name_by_id_before[d] for d in duplicate_ids} == DUPLICATE_NAMES

    await consolidate_entities_pipeline(similarity_threshold=0.6, dataset=DATASET, user=user)

    async with set_database_global_context_variables(target.id, target.owner_id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        nodes_after, edges_after = await graph.get_graph_data()
        entities_after = {
            str(node_id): props for node_id, props in nodes_after if props.get("type") == "Entity"
        }

        # Exactly one of the duplicate pair remains.
        survivors = [
            node_id
            for node_id, props in entities_after.items()
            if props.get("name") in DUPLICATE_NAMES
        ]
        assert len(survivors) == 1, (
            f"expected a single survivor, found {[entities_after[s]['name'] for s in survivors]}"
        )
        survivor_id = survivors[0]

        # All original edges are preserved on the survivor (both directions /
        # both the canonical-side and the duplicate-side edge).
        name_by_id = {str(node_id): props.get("name") for node_id, props in nodes_after}
        survivor_rels = {
            (rel, name_by_id.get(str(target_id)))
            for source_id, target_id, rel, _ in edges_after
            if str(source_id) == survivor_id
        }
        assert ("located_in", "United States") in survivor_rels
        assert ("contains", "Central Park") in survivor_rels

        # Neighbors are untouched (no accidental over-merge).
        names_after = {props.get("name") for props in entities_after.values()}
        assert "United States" in names_after
        assert "Central Park" in names_after

        # The duplicate node is gone, and so is its name embedding: embed the
        # deleted name and confirm its id is not among the nearest matches.
        deleted_ids = duplicate_ids - {survivor_id}
        assert deleted_ids and all(d not in entities_after for d in deleted_ids)
        for deleted_id in deleted_ids:
            probe_vector = (await vector.embed_data([name_by_id_before[deleted_id]]))[0]
            results = await vector.search(
                "Entity_name", query_text=None, query_vector=probe_vector, limit=50
            )
            present_ids = {str(getattr(result, "id", None)) for result in results}
            assert deleted_id not in present_ids
