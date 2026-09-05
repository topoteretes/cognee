"""LLM-free knowledge graph + summaries with GLiNER (SDK-537).

The default ``cognee.cognify()`` extracts the graph and the chunk summaries with
an LLM. This example builds both with a local GLiNER2 model instead: no
``extract_content_graph`` and no ``extract_summary`` calls are made. Embeddings
in ``add_data_points`` still run, so an embedding provider must be configured.

Requirements::

    pip install "cognee[gliner]"

The first run downloads ``fastino/gliner2.5-base-v1`` (~800 MB) into the
Hugging Face cache.

What it prints: the schema GLiNER was given (caller labels here; drop the
``entity_types``/``relation_types`` arguments to exercise the ontology and
label-bank fallbacks), the node/edge/summary counts in the stored graph, and
kept-vs-dropped relation counts so endpoint-resolution loss is measured rather
than assumed.
"""

import asyncio
import os

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.users.methods import get_default_user
from cognee.tasks.graph.gliner import GlinerRunStats, get_gliner_tasks

TEXT = """
Tim Cook is the chief executive officer of Apple Inc., headquartered in Cupertino,
California. Before joining Apple in 1998 he worked at Compaq and IBM. Apple was
founded by Steve Jobs, Steve Wozniak and Ronald Wayne in 1976 and today produces
the iPhone, the Mac and the Apple Watch. In 2014 Apple acquired Beats Electronics,
the headphone company co-founded by Dr. Dre and Jimmy Iovine, for three billion
dollars. Apple Park, the company's campus, opened in 2017.
"""

DATASET = "gliner_demo"


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(TEXT, dataset_name=DATASET)
    user = await get_default_user()

    stats = GlinerRunStats()
    tasks = await get_gliner_tasks(
        entity_types={
            "person": "Full name of a human being",
            "organization": "Company or institution",
            "location": "City, region, campus or country",
            "product": "Commercial product",
            "date": "Year or calendar date",
            "money": "Monetary amount",
        },
        relation_types=["works_for", "headquartered_in", "founded_by", "produces", "acquired"],
        stats=stats,
    )
    # skip_connection_test: the first-run environment check probes the LLM as
    # well; this pipeline never calls one.
    await cognee.run_custom_pipeline(
        tasks=tasks,
        user=user,
        dataset=DATASET,
        pipeline_name="cognify_pipeline",
        skip_connection_test=True,
    )

    schema = stats.schema
    print(f"\nschema source: {schema.source}")
    print(f"entity types:   {sorted(schema.entity_types)}")
    print(f"relation types: {sorted(schema.relation_types)}")
    print(f"\nchunks processed: {stats.chunks}")
    print(f"nodes mapped:     {stats.nodes}")
    print(
        f"relations: {stats.candidate_edges} proposed, "
        f"{stats.kept_edges} kept, {stats.dropped_edges} dropped (endpoint did not resolve)"
    )

    datasets = await cognee.datasets.list_datasets(user)
    dataset = next(d for d in datasets if d.name == DATASET)
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()

    by_type: dict[str, int] = {}
    for _node_id, props in nodes:
        by_type[props.get("type", "?")] = by_type.get(props.get("type", "?"), 0) + 1
    print(f"\nstored graph: {len(nodes)} nodes, {len(edges)} edges")
    for type_name, count in sorted(by_type.items()):
        print(f"  {type_name}: {count}")

    summaries = [props for _id, props in nodes if props.get("type") == "TextSummary"]
    print(f"\nTextSummary nodes: {len(summaries)}")
    for props in summaries:
        print("  ---")
        print("  " + (props.get("text") or "<empty>").replace("\n", "\n  "))

    structural = {"contains", "is_a", "is_part_of", "made_from", "belongs_to_set"}
    entity_edges = sorted(
        {(str(edge[0]), edge[2], str(edge[1])) for edge in edges if edge[2] not in structural}
    )
    print(f"\nentity relations stored: {len(entity_edges)}")
    names = {str(node_id): props.get("name") for node_id, props in nodes}
    for source, relation, target in entity_edges:
        print(f"  {names.get(source, source)} --{relation}--> {names.get(target, target)}")

    if os.getenv("LLM_API_KEY"):
        from cognee import SearchType

        results = await cognee.search(
            query_text="Who leads Apple and where is it based?",
            query_type=SearchType.SUMMARIES,
            datasets=[DATASET],
            top_k=3,
        )
        print("\nSUMMARIES search (per dataset):")
        for per_dataset in results:
            for item in per_dataset.get("search_result", []):
                print("  -", (item.get("text") or "<empty>").replace("\n", " | "))


if __name__ == "__main__":
    asyncio.run(main())
