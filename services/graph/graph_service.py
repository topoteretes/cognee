import json
import os
from datetime import datetime
import logging
from services.timeline.events import get_events

logger = logging.getLogger("graph_service")
TIME_MAP_PATH = os.path.join(os.getcwd(), "node_creation_times.json")

def load_creation_times() -> dict:
    """Loads the creation timestamps for nodes and edges."""
    if os.path.exists(TIME_MAP_PATH):
        try:
            with open(TIME_MAP_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"nodes": {}, "edges": {}}

def save_creation_times(time_map: dict):
    """Saves the creation timestamps to file."""
    try:
        with open(TIME_MAP_PATH, "w") as f:
            json.dump(time_map, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save creation times: {e}")

async def get_raw_graph():
    """Fetches raw nodes and edges from Cognee's graph engine."""
    # We do a dynamic import here to make sure settings are applied first
    from services.memory.cognee_service import configure_cognee
    configure_cognee()
    
    from cognee.infrastructure.databases.graph import get_graph_engine
    try:
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        return nodes, edges
    except Exception as e:
        logger.warning(f"Could not retrieve graph from engine (might be empty): {e}")
        return [], []

async def sync_creation_times():
    """Syncs current nodes/edges with their first-seen timestamp."""
    nodes, edges = await get_raw_graph()
    time_map = load_creation_times()
    changed = False
    now_str = datetime.utcnow().isoformat() + "Z"
    
    # Sync nodes
    for node_id, _ in nodes:
        node_key = str(node_id)
        if node_key not in time_map["nodes"]:
            time_map["nodes"][node_key] = now_str
            changed = True
            
    # Sync edges
    for source_id, target_id, rel_name, _ in edges:
        edge_key = f"{source_id}-{target_id}-{rel_name}"
        if edge_key not in time_map["edges"]:
            time_map["edges"][edge_key] = now_str
            changed = True
            
    if changed:
        save_creation_times(time_map)
        
    return time_map

async def get_graph_data(target_timestamp: str = None) -> dict:
    """Returns formatted nodes and edges, optionally filtered for a Time Machine snapshot."""
    # Make sure timestamps are synchronized
    time_map = await sync_creation_times()
    nodes, edges = await get_raw_graph()
    
    # Compute recall frequency for heatmap
    recent_recalls = get_events(limit=20, event_type="RecallTriggered")
    recall_hits = {}
    for ev in recent_recalls:
        query_text = ev.get("metadata", {}).get("query", "").lower()
        # Find which nodes match words in the query
        for node_id, props in nodes:
            node_name = str(props.get("name") or props.get("label") or node_id).lower()
            if node_name in query_text:
                recall_hits[str(node_id)] = recall_hits.get(str(node_id), 0) + 1
                
    # Filter nodes and edges by timestamp if Time Machine is active
    if target_timestamp:
        try:
            target_dt = datetime.fromisoformat(target_timestamp.replace("Z", "+00:00"))
        except Exception:
            target_dt = None
            
        if target_dt:
            filtered_nodes = []
            valid_node_ids = set()
            for node_id, props in nodes:
                node_key = str(node_id)
                created_str = time_map["nodes"].get(node_key)
                if created_str:
                    created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if created_dt <= target_dt:
                        filtered_nodes.append((node_id, props))
                        valid_node_ids.add(node_id)
                else:
                    # Default keep if not mapped yet
                    filtered_nodes.append((node_id, props))
                    valid_node_ids.add(node_id)
                    
            filtered_edges = []
            for source_id, target_id, rel_name, props in edges:
                edge_key = f"{source_id}-{target_id}-{rel_name}"
                created_str = time_map["edges"].get(edge_key)
                if created_str:
                    created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if created_dt <= target_dt and source_id in valid_node_ids and target_id in valid_node_ids:
                        filtered_edges.append((source_id, target_id, rel_name, props))
                else:
                    if source_id in valid_node_ids and target_id in valid_node_ids:
                        filtered_edges.append((source_id, target_id, rel_name, props))
                        
            nodes = filtered_nodes
            edges = filtered_edges

    # Format nodes
    formatted_nodes = []
    total_nodes = len(nodes)
    
    # Calculate degrees for centrality
    degrees = {}
    for source_id, target_id, _, _ in edges:
        degrees[str(source_id)] = degrees.get(str(source_id), 0) + 1
        degrees[str(target_id)] = degrees.get(str(target_id), 0) + 1
        
    for node_id, props in nodes:
        node_key = str(node_id)
        
        # Calculate Heatmap Score (0.0 to 1.0)
        hits = recall_hits.get(node_key, 0)
        # Heat decays or scales
        heat = min(1.0, hits / 5.0)
        
        # Calculate DNA Properties
        deg = degrees.get(node_key, 0)
        importance = min(1.0, (deg / max(1, total_nodes - 1)))
        
        created_at_str = time_map["nodes"].get(node_key, datetime.utcnow().isoformat() + "Z")
        try:
            created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            age_seconds = (datetime.utcnow().replace(tzinfo=created_dt.tzinfo) - created_dt).total_seconds()
            # Freshness decays over time (e.g. 1.0 decays to 0.1 over a few days)
            freshness = 1.0 / (1.0 + (age_seconds / 86400.0))
        except Exception:
            freshness = 1.0
            age_seconds = 0
            
        trust = float(props.get("trust", 0.9))
        
        dna = {
            "importance": round(importance, 2),
            "freshness": round(freshness, 2),
            "trust": round(trust, 2),
            "frequency": hits,
            "age_seconds": int(age_seconds),
            "connections": deg
        }
        
        formatted_nodes.append({
            "id": node_key,
            "label": props.get("name") or props.get("label") or props.get("text", node_key),
            "type": props.get("type") or props.get("node_type", "Concept"),
            "properties": props,
            "heat": round(heat, 2),
            "dna": dna
        })
        
    # Format edges
    formatted_edges = []
    for source_id, target_id, rel_name, props in edges:
        formatted_edges.append({
            "source": str(source_id),
            "target": str(target_id),
            "label": rel_name,
            "properties": props
        })
        
    return {
        "nodes": formatted_nodes,
        "edges": formatted_edges,
        "metrics": {
            "total_nodes": len(formatted_nodes),
            "total_edges": len(formatted_edges),
            "density": round(len(formatted_edges) / max(1, len(formatted_nodes)), 2)
        }
    }
