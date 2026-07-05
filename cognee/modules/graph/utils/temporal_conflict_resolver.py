from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

class TemporalConflictResolver:
    """
    Handles resolution of temporal contradictions between incoming graph edges 
    and existing state based on metadata, timestamps, or sequential ingestion.
    """
    
    @staticmethod
    def resolve_conflicts(
        incoming_edges: List[Dict[str, Any]], 
        existing_edges: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compares incoming edges against known edges.
        Detects if a new edge updates or supersedes an old relationship.
        
        Returns:
        --------
        Tuple[List[Dict], List[Dict]]: (edges_to_upsert, edges_to_update_as_superseded)
        """
        if not existing_edges:
            # If no historical state is provided yet, pass through incoming data cleanly
            return incoming_edges, []

        edges_to_upsert = []
        edges_to_update_as_superseded = []

        # Create a lookup map for existing edges by logical identity: (source, relationship)
        existing_map = {
            (edge.get("source_node_id"), edge.get("relationship_name")): edge
            for edge in existing_edges
        }

        for incoming in incoming_edges:
            key = (incoming.get("source_node_id"), incoming.get("relationship_name"))
            
            if key in existing_map:
                existing = existing_map[key]
                
                # Check if the destination has changed (a semantic/structural contradiction)
                if existing.get("destination_node_id") != incoming.get("destination_node_id"):
                    # Concrete resolution strategy: Mark incoming as current, old as superseded
                    updated_attributes = dict(existing.get("attributes", {}))
                    updated_attributes["status"] = "superseded"
                    
                    existing_copy = dict(existing)
                    existing_copy["attributes"] = updated_attributes
                    edges_to_update_as_superseded.append(existing_copy)
                    
                    # Tag incoming as active/current
                    incoming_attributes = dict(incoming.get("attributes", {}))
                    incoming_attributes["status"] = "current"
                    incoming["attributes"] = incoming_attributes
                    
                edges_to_upsert.append(incoming)
            else:
                edges_to_upsert.append(incoming)

        return edges_to_upsert, edges_to_update_as_superseded