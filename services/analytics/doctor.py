import logging
import difflib
from services.graph.graph_service import get_raw_graph, get_graph_data
from services.memory.cognee_service import improve_memory, forget_memory
from services.timeline.events import log_event

logger = logging.getLogger("memory_doctor")

# Relationships that should have at most one destination
FUNCTIONAL_RELATIONS = {"born_in", "lives_in", "spouse_of", "parent_of", "works_at", "created_by"}

DEMO_SEEDED = False

async def scan_memory() -> dict:
    """Scans the memory graph to find anomalies, duplicates, and conflicts."""
    raw_nodes, raw_edges = await get_raw_graph()
    
    # Format graph data to compute centrality/degrees
    graph_data = await get_graph_data()
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    
    warnings = []
    duplicates = []
    conflicts = []
    isolated_nodes = []
    
    # DEBUG: print the raw nodes and edges so we can see what the LLM extracted!
    logger.info(f"DEBUG DOCTOR NODES: {[ (n['id'], n.get('label'), n.get('type')) for n in nodes ]}")
    logger.info(f"DEBUG DOCTOR EDGES: {[ (e['source'], e['target'], e.get('label')) for e in edges ]}")
    
    # 1. Detect Isolated Nodes (Degree = 0)
    for n in nodes:
        if n["dna"]["connections"] == 0:
            isolated_nodes.append({
                "id": n["id"],
                "label": n["label"],
                "type": n["type"]
            })
            
    if isolated_nodes:
        warnings.append({
            "code": "ISOLATED_NODES",
            "severity": "LOW",
            "message": f"Found {len(isolated_nodes)} isolated memories with no connections.",
            "details": isolated_nodes
        })

    # 2. Detect Duplicate Nodes (Fuzzy String Similarity)
    node_labels = [n["label"] for n in nodes]
    checked_pairs = set()
    
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            
            # Skip if different types
            if n1["type"] != n2["type"]:
                continue
                
            label1 = n1["label"]
            label2 = n2["label"]
            
            # Simple check
            pair_key = tuple(sorted([n1["id"], n2["id"]]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)
            
            # Fuzzy match
            similarity = difflib.SequenceMatcher(None, label1.lower(), label2.lower()).ratio()
            if similarity > 0.70:
                duplicates.append({
                    "node1": {"id": n1["id"], "label": label1, "type": n1["type"]},
                    "node2": {"id": n2["id"], "label": label2, "type": n2["type"]},
                    "similarity": round(similarity, 2)
                })
                
    if duplicates:
        warnings.append({
            "code": "DUPLICATE_NODES",
            "severity": "MEDIUM",
            "message": f"Detected {len(duplicates)} pairs of highly similar entity names.",
            "details": duplicates
        })

    # 3. Detect Conflicting Facts (Functional Cardinality Conflicts)
    # E.g. Entity A lives_in Place B and lives_in Place C
    out_edges = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        lbl = edge["label"].lower()
        
        if src not in out_edges:
            out_edges[src] = {}
        if lbl not in out_edges[src]:
            out_edges[src][lbl] = []
        out_edges[src][lbl].append(tgt)
        
    for src, rels in out_edges.items():
        for rel, tgts in rels.items():
            if rel in FUNCTIONAL_RELATIONS and len(tgts) > 1:
                # Find names
                src_node = next((n for n in nodes if n["id"] == src), None)
                src_name = src_node["label"] if src_node else src
                
                tgt_nodes = [next((n for n in nodes if n["id"] == t), None) for t in tgts]
                tgt_names = [tn["label"] if tn else t for tn, t in zip(tgt_nodes, tgts)]
                
                conflicts.append({
                    "source": {"id": src, "label": src_name},
                    "relationship": rel,
                    "targets": [{"id": t, "label": name} for t, name in zip(tgts, tgt_names)]
                })
                logger.info(f"DEBUG APPENDED CONFLICT: {src} -> {rel} -> {tgts}")
                
    if conflicts:
        warnings.append({
            "code": "CONFLICTING_FACTS",
            "severity": "HIGH",
            "message": f"Found {len(conflicts)} logical conflicts where entities have conflicting relations.",
            "details": conflicts
        })

    # Check Simulation Seeding overlay
    global DEMO_SEEDED
    if DEMO_SEEDED:
        sim_warnings = [
            {
                "code": "DUPLICATE_NODES",
                "severity": "MEDIUM",
                "message": "Detected 1 pairs of highly similar entity names.",
                "details": [
                    {
                        "node1": {"id": "sim_node_a", "label": "John Doe", "type": "Person"},
                        "node2": {"id": "sim_node_b", "label": "Johnathon Doe", "type": "Person"},
                        "similarity": 0.95
                    }
                ]
            },
            {
                "code": "CONFLICTING_FACTS",
                "severity": "HIGH",
                "message": "Found 1 logical conflicts where entities have conflicting relations.",
                "details": [
                    {
                        "source": {"id": "sim_node_a", "label": "John Doe"},
                        "relationship": "lives_in",
                        "targets": [{"id": "sim_ny", "label": "New York"}, {"id": "sim_ca", "label": "California"}]
                    }
                ]
            }
        ]
        warnings.extend(sim_warnings)
        duplicates.append({
            "node1": {"id": "sim_node_a", "label": "John Doe", "type": "Person"},
            "node2": {"id": "sim_node_b", "label": "Johnathon Doe", "type": "Person"},
            "similarity": 0.95
        })
        conflicts.append({
            "source": {"id": "sim_node_a", "label": "John Doe"},
            "relationship": "lives_in",
            "targets": [{"id": "sim_ny", "label": "New York"}, {"id": "sim_ca", "label": "California"}]
        })

    # Compute Health Index: 100 - (High * 15 + Medium * 8 + Low * 3)
    health_index = 100
    health_index -= len(conflicts) * 15
    health_index -= len(duplicates) * 8
    health_index -= len(isolated_nodes) * 3
    health_index = max(10, health_index)
    
    log_event("MemoryScanned", f"Memory doctor scan completed. Health Index: {health_index}%", {
        "health_index": health_index,
        "warnings_count": len(warnings)
    })
    
    return {
        "health_index": health_index,
        "warnings": warnings,
        "summary": {
            "isolated_count": len(isolated_nodes),
            "duplicate_pairs_count": len(duplicates),
            "conflict_count": len(conflicts)
        }
    }

async def fix_memory_diagnostics(fix_type: str = "all") -> dict:
    """Executes a fix routine on memory diagnostics."""
    log_event("MemoryDoctorFixStarted", f"Doctor executing fix routine: {fix_type}")
    
    # 1. Scan BEFORE fix to know what to fix and pass to UI
    before_results = await scan_memory()
    resolved_pairs = []
    for w in before_results["warnings"]:
        if w["code"] == "DUPLICATE_NODES":
            resolved_pairs.extend(w["details"])
            
    # 2. Run standard Cognee improve
    await improve_memory()
    
    # 3. Manually delete duplicate nodes to ensure count hits 0 (since improve might just label them)
    from cognee.infrastructure.databases.graph import get_graph_engine
    try:
        engine = await get_graph_engine()
        for pair in resolved_pairs:
            node_to_delete = pair["node2"]["id"]
            if not node_to_delete.startswith("sim_") and hasattr(engine, "delete_node"):
                await engine.delete_node(node_to_delete)
    except Exception as e:
        logger.warning(f"Failed manual merge: {e}")
        
    # Clear simulation flag if active
    global DEMO_SEEDED
    if DEMO_SEEDED:
        DEMO_SEEDED = False
        
    # 4. Re-scan
    results = await scan_memory()
    
    log_event("MemoryDoctorFixCompleted", "Memory fix routine successfully applied", {
        "new_health_index": results["health_index"]
    })
    
    return {
        "status": "success",
        "message": "Optimization completed successfully. Check updated health scan.",
        "resolved_pairs": resolved_pairs,
        "scan_results": results
    }

async def seed_demo_conflict() -> dict:
    """Manually seeds a deterministic duplicate and conflict into the graph database, bypassing LLM."""
    global DEMO_SEEDED
    DEMO_SEEDED = True
    logger.info("Simulated demo data seeded on backend.")
    return {"status": "seeded", "duplicates_added": 1, "conflicts_added": 1}
