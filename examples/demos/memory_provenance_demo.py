"""Demo: REAL memory-provenance projection (relational → graph).

This drives the *production* projection function
``cognee.api.v1.visualize.memory_provenance.build_provenance_graph`` — the same
code path ``get_memory_provenance_graph()`` uses to read live Tenants/Users/
Agents/Datasets/Data/Sessions from the relational database.

It also runs the live reader against the actual DB and prints a summary, so you
can see the real projection working end-to-end (no knowledge-graph DB / LLM).

The representative records below mirror the dict shapes the live reader yields.
"""

import asyncio
import os

import cognee
from cognee.api.v1.visualize.memory_provenance import build_provenance_graph
from cognee.modules.visualization.preprocessor import preprocess
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)

DEST = os.path.join(os.path.expanduser("~"), "cognee_provenance_demo.html")


def representative_records():
    tenants = [{"id": "acme", "name": "Acme Logistics"}]
    users = [
        {"id": "alice", "name": "Alice Chen", "tenant_ids": ["acme"]},
        {"id": "bob", "name": "Bob Ramirez", "tenant_ids": ["acme"]},
        {"id": "carol", "name": "Carol Davis", "tenant_ids": ["acme"]},
    ]
    datasets = [
        {"id": "fleet", "name": "Fleet Ops", "owner_id": "alice", "tenant_id": "acme"},
        {"id": "carrier", "name": "Carrier Intel", "owner_id": "bob", "tenant_id": "acme"},
        {"id": "driver", "name": "Driver Records", "owner_id": "alice", "tenant_id": "acme"},
        {"id": "billing", "name": "Billing & Claims", "owner_id": "carol", "tenant_id": "acme"},
    ]
    files = [
        {"id": "f_carlos", "name": "carlos_interview.txt", "dataset_ids": ["fleet"]},
        {"id": "f_log", "name": "dispatch_log_0521.csv", "dataset_ids": ["fleet"]},
        {"id": "f_echo", "name": "echo_profile.pdf", "dataset_ids": ["carrier"]},
        {"id": "f_landstar", "name": "landstar_contract.pdf", "dataset_ids": ["carrier"]},
        {"id": "f_mika", "name": "mika_driver_file.txt", "dataset_ids": ["driver"]},
        {"id": "f_priya", "name": "priya_review.txt", "dataset_ids": ["driver"]},
        {"id": "f_inv", "name": "may_invoices.csv", "dataset_ids": ["billing"]},
    ]
    agents = [
        {
            "id": "dispatch",
            "name": "dispatch-copilot",
            "user_id": "alice",
            "session_id": "s1",
            "datasets": [
                {"dataset_id": "fleet", "role": "read_write"},
                {"dataset_id": "carrier", "role": "read"},
            ],
        },
        {
            "id": "research",
            "name": "carrier-research-agent",
            "user_id": "bob",
            "session_id": "s3",
            "datasets": [
                {"dataset_id": "carrier", "role": "read_write"},
                {"dataset_id": "driver", "role": "read"},
            ],
        },
        {
            "id": "audit",
            "name": "compliance-auditor",
            "user_id": "carol",
            "session_id": "s4",
            "datasets": [
                {"dataset_id": "billing", "role": "read_write"},
                {"dataset_id": "driver", "role": "read"},
            ],
        },
    ]
    sessions = [
        {"id": "s1", "name": "sess_0521_a1f3", "user_id": "alice", "dataset_id": "fleet"},
        {
            "id": "s2",
            "name": "sess_0521_b7c2",
            "user_id": "alice",
            "dataset_id": "fleet",
            "agent_id": "dispatch",
        },
        {"id": "s3", "name": "sess_0522_d4e9", "user_id": "bob", "dataset_id": "carrier"},
        {"id": "s4", "name": "sess_0523_f1a8", "user_id": "carol", "dataset_id": "billing"},
        {
            "id": "s5",
            "name": "sess_0524_3b6d",
            "user_id": "carol",
            "dataset_id": "driver",
            "agent_id": "audit",
        },
    ]

    # Memory layer (entities/types) linked back to the files they came from.
    entities = {
        "e_carlos": ("Carlos", "Person", "f_carlos"),
        "e_mika": ("Mika", "Person", "f_mika"),
        "e_sandra": ("Sandra", "Person", "f_carlos"),
        "e_priya": ("Priya", "Person", "f_priya"),
        "e_echo": ("Echo Global Logistics", "Broker", "f_echo"),
        "e_landstar": ("Landstar", "Broker", "f_landstar"),
        "e_loadsearch": ("load-search", "Tool", "f_carlos"),
    }
    mem_nodes = [
        ("et:" + t, {"type": "EntityType", "name": t}) for t in ("Person", "Broker", "Tool")
    ]
    mem_edges = []
    mem_links = []
    for eid, (ename, etype, src) in entities.items():
        mem_nodes.append((eid, {"type": "Entity", "name": ename}))
        mem_edges.append((eid, "et:" + etype, "is_a", {}))
        mem_links.append({"node_id": eid, "data_id": src, "dataset_id": None})
    mem_edges += [
        ("e_carlos", "e_echo", "works_at", {}),
        ("e_mika", "e_landstar", "works_at", {}),
        ("e_carlos", "e_loadsearch", "uses", {}),
        ("e_sandra", "e_carlos", "interviewed", {}),
    ]
    memory = {"nodes": mem_nodes, "edges": mem_edges, "links": mem_links}

    return dict(
        tenants=tenants,
        users=users,
        datasets=datasets,
        files=files,
        agents=agents,
        sessions=sessions,
        memory=memory,
    )


async def main():
    # 1) The REAL projection function, fed representative records.
    graph_data = build_provenance_graph(**representative_records())

    pre = preprocess(graph_data)
    type_nodes = [n for n in pre.schema_graph["nodes"] if n.get("type") == "GraphNodeType"]
    type_nodes.sort(key=lambda n: (n.get("rank", 99), -(n.get("instance_count") or 0)))
    print("\n=== Memory provenance — levels (left → right) ===")
    for n in type_nodes:
        samples = ", ".join(n.get("samples", []))
        print(f"  [{n.get('rank')!s:>4}] {n['name']}: {n.get('instance_count')} — {samples}")

    await cognee_network_visualization(graph_data, DEST)
    print(f"\n=== Visualization written to: {DEST} ===")

    # 2) The SAME projection, reading live relational data.
    live_nodes, live_edges = await cognee.get_memory_provenance_graph()
    from collections import Counter

    counts = Counter(props.get("type") for _, props in live_nodes)
    print("\n=== Live relational projection (real DB) ===")
    print(f"  {dict(counts)} | {len(live_edges)} edges")
    print("  (sparse until more tenants/users/agents/sessions exist in the DB)")


if __name__ == "__main__":
    asyncio.run(main())
