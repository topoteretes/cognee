# cognee/tasks/memify/extract_usage_frequency.py
from typing import List, Dict, Any
from datetime import datetime, timedelta
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.pipelines.tasks.task import Task

async def extract_usage_frequency(
    subgraphs: List[CogneeGraph], 
    time_window: timedelta = timedelta(days=7),
    min_interaction_threshold: int = 1
) -> Dict[str, Any]:
    """
    Extract usage frequency from CogneeUserInteraction nodes
    
    :param subgraphs: List of graph subgraphs
    :param time_window: Time window to consider for interactions
    :param min_interaction_threshold: Minimum interactions to track
    :return: Dictionary of usage frequencies
    """
    current_time = datetime.now()
    node_frequencies = {}
    edge_frequencies = {}
    
    for subgraph in subgraphs:
        # Filter CogneeUserInteraction nodes within time window
        user_interactions = [
            interaction for interaction in subgraph.nodes 
            if (interaction.get('type') == 'CogneeUserInteraction' and 
                current_time - datetime.fromisoformat(interaction.get('timestamp', current_time.isoformat())) <= time_window)
        ]
        
        # Count node and edge frequencies
        for interaction in user_interactions:
            target_node_id = interaction.get('target_node_id')
            edge_type = interaction.get('edge_type')
            
            if target_node_id:
                node_frequencies[target_node_id] = node_frequencies.get(target_node_id, 0) + 1
            
            if edge_type:
                edge_frequencies[edge_type] = edge_frequencies.get(edge_type, 0) + 1
    
    # Filter frequencies above threshold
    filtered_node_frequencies = {
        node_id: freq for node_id, freq in node_frequencies.items() 
        if freq >= min_interaction_threshold
    }
    
    filtered_edge_frequencies = {
        edge_type: freq for edge_type, freq in edge_frequencies.items() 
        if freq >= min_interaction_threshold
    }
    
    return {
        'node_frequencies': filtered_node_frequencies,
        'edge_frequencies': filtered_edge_frequencies,
        'last_processed_timestamp': current_time.isoformat()
    }

async def add_frequency_weights(
    graph_adapter, 
    usage_frequencies: Dict[str, Any]
) -> None:
    """
    Add frequency weights to graph nodes and edges
    
    :param graph_adapter: Graph database adapter
    :param usage_frequencies: Calculated usage frequencies
    """
    # Update node frequencies
    for node_id, frequency in usage_frequencies['node_frequencies'].items():
        try:
            node = graph_adapter.get_node(node_id)
            if node:
                node_properties = node.get_properties() or {}
                node_properties['frequency_weight'] = frequency
                graph_adapter.update_node(node_id, node_properties)
        except Exception as e:
            print(f"Error updating node {node_id}: {e}")
    
    # Note: Edge frequency update might require backend-specific implementation
    print("Edge frequency update might need backend-specific handling")

def usage_frequency_pipeline_entry(graph_adapter):
    """
    Memify pipeline entry for usage frequency tracking
    
    :param graph_adapter: Graph database adapter
    :return: Usage frequency results
    """
    extraction_tasks = [
        Task(extract_usage_frequency, 
             time_window=timedelta(days=7), 
             min_interaction_threshold=1)
    ]
    
    enrichment_tasks = [
        Task(add_frequency_weights, task_config={"batch_size": 1})
    ]
    
    return extraction_tasks, enrichment_tasks