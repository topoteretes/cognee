"""Print every Rule node in the human_memory graph, separating
seed rules (source=alex_playbook) from agent-approved ones
(source=agent_proposal:*). Sanity check for the review CLI."""

import asyncio

import cognee
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user


async def _inspect(_data, ctx: PipelineContext = None):
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _edges = await graph.get_graph_data()

    rules = [n for n in nodes if isinstance(n, tuple) and "rule_id" in (n[1] or {})]
    rules.sort(key=lambda n: (n[1].get("source", ""), n[1].get("rule_id", "")))

    print(f"\n{'=' * 78}")
    print(f"Rules in human_memory graph: {len(rules)}")
    print("=" * 78)

    by_source: dict[str, list] = {}
    for _nid, props in rules:
        by_source.setdefault(props.get("source", "?"), []).append(props)

    for source, items in sorted(by_source.items()):
        print(f"\n[{source}]  ({len(items)} rule(s))")
        for r in items:
            node_set = r.get("belongs_to_set") or r.get("node_set") or "-"
            print(f"  {r.get('rule_id'):<45} status={r.get('status')}  node_set={node_set}")
            print(f"    trigger: {(r.get('trigger') or '').strip()[:100]}")
            print(f"    action : {(r.get('action') or '').strip()[:100]}")


async def main() -> None:
    user = await get_default_user()
    await cognee.run_custom_pipeline(
        tasks=[Task(_inspect)],
        data=[None],
        dataset="human_memory",
        user=user,
        pipeline_name="inspect_rulebook",
    )


if __name__ == "__main__":
    asyncio.run(main())
