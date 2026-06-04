"""Demo: schema & entity inventory + visualization (no DB / no LLM required).

Builds a representative freight-broker graph in the exact shape that
``GraphDBInterface.get_graph_data()`` returns, then drives the real
visualization code:
  * preprocess()                  -> PR2 schema payload (samples + relationships)
  * cognee_network_visualization  -> HTML with the PR3 schema side panel

This renders exactly what a live ``cognify`` would feed the visualizer; only
the LLM-extraction step is stood in with a fixed graph so the demo is fast and
deterministic.
"""

import asyncio
import os

from cognee.modules.visualization.preprocessor import preprocess
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)

DEST = os.path.join(os.path.expanduser("~"), "cognee_schema_demo.html")

_META = {
    "updated_at": 123,
    "created_at": 123,
    "source_task": "extract_graph_from_data",
    "source_pipeline": "cognify_pipeline",
    "source_node_set": "freight_demo",
    "source_user": "demo@cognee.ai",
}


def _entity(node_id, name):
    return (node_id, {"type": "Entity", "name": name, **_META})


def _entity_type(node_id, name):
    return (node_id, {"type": "EntityType", "name": name, **_META})


def build_graph():
    # EntityType meta-nodes (the semantic types Person/Broker/Tool).
    nodes = [
        _entity_type("t:person", "Person"),
        _entity_type("t:broker", "Broker"),
        _entity_type("t:tool", "Tool"),
        # Person instances
        _entity("e:carlos", "Carlos"),
        _entity("e:mika", "Mika"),
        _entity("e:sandra", "Sandra"),
        _entity("e:priya", "Priya"),
        # Broker instances
        _entity("e:echo", "Echo Global Logistics"),
        _entity("e:landstar", "Landstar"),
        # Tool instance
        _entity("e:loadsearch", "load-search"),
    ]

    edges = []
    # is_a edges: every Entity -> its EntityType (this is what PR2 resolves so
    # the schema shows Person/Broker/Tool instead of one giant "Entity" box).
    for eid, tid in [
        ("e:carlos", "t:person"),
        ("e:mika", "t:person"),
        ("e:sandra", "t:person"),
        ("e:priya", "t:person"),
        ("e:echo", "t:broker"),
        ("e:landstar", "t:broker"),
        ("e:loadsearch", "t:tool"),
    ]:
        edges.append((eid, tid, "is_a", {}))

    # Inter-entity relationships (what the system "understood" from the text).
    edges += [
        ("e:carlos", "e:echo", "works_at", {}),
        ("e:mika", "e:landstar", "works_at", {}),
        ("e:carlos", "e:loadsearch", "uses", {}),
        ("e:mika", "e:loadsearch", "uses", {}),
        ("e:priya", "e:loadsearch", "manages", {}),
        ("e:sandra", "e:carlos", "interviewed", {}),
        ("e:sandra", "e:priya", "interviewed", {}),
    ]
    return (nodes, edges)


async def main():
    graph_data = build_graph()

    # PR2: the schema payload the side panel consumes.
    pre = preprocess(graph_data)
    type_nodes = [n for n in pre.schema_graph["nodes"] if n.get("type") == "GraphNodeType"]
    type_nodes.sort(key=lambda n: (-n.get("instance_count", 0), n.get("name") or ""))

    print("\n=== Schema & Entity Inventory (what the panel shows) ===")
    for n in type_nodes:
        count = n.get("instance_count", 0)
        samples = n.get("samples", [])
        more = f" (+{count - len(samples)} more)" if count > len(samples) else ""
        print(f"  {n['name']}: {count} — {', '.join(samples)}{more}")
        for rel in n.get("relationships", [])[:4]:
            print(f"      ↳ {rel['relation']} → {rel['to_type']} ({rel['count']})")

    # PR3: render the HTML (includes the schema side panel + highlight bridge).
    await cognee_network_visualization(graph_data, DEST)
    print(f"\n=== Visualization written to: {DEST} ===")


if __name__ == "__main__":
    asyncio.run(main())
