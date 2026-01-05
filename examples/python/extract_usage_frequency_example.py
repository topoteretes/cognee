# cognee/examples/usage_frequency_example.py
"""
End-to-end example demonstrating usage frequency tracking in Cognee.

This example shows how to:
1. Add data and build a knowledge graph
2. Run searches with save_interaction=True to track usage
3. Extract and apply frequency weights using the memify pipeline
4. Query and analyze the frequency data

The frequency weights can be used to:
- Rank frequently referenced entities higher during retrieval
- Adjust scoring for completion strategies
- Expose usage metrics in dashboards or audits
"""
import asyncio
from datetime import timedelta
from typing import List

import cognee
from cognee.api.v1.search import SearchType
from cognee.tasks.memify.extract_usage_frequency import (
    create_usage_frequency_pipeline,
    run_usage_frequency_update,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.shared.logging_utils import get_logger

logger = get_logger("usage_frequency_example")


async def setup_knowledge_base():
    """Set up a fresh knowledge base with sample data."""
    logger.info("Setting up knowledge base...")
    
    # Reset cognee state for clean slate
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Sample conversation about AI/ML topics
    conversation = [
        "Alice discusses machine learning algorithms and their applications in computer vision.",
        "Bob asks about neural networks and how they differ from traditional algorithms.",
        "Alice explains deep learning concepts including CNNs and transformers.",
        "Bob wants more details about neural networks and backpropagation.",
        "Alice describes reinforcement learning and its use in robotics.",
        "Bob inquires about natural language processing and transformers.",
    ]

    # Add conversation data and build knowledge graph
    logger.info("Adding conversation data...")
    await cognee.add(conversation, dataset_name="ai_ml_conversation")
    
    logger.info("Building knowledge graph (cognify)...")
    await cognee.cognify()
    
    logger.info("Knowledge base setup complete")


async def simulate_user_searches():
    """Simulate multiple user searches to generate interaction data."""
    logger.info("Simulating user searches with save_interaction=True...")
    
    # Different queries that will create CogneeUserInteraction nodes
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "Tell me about deep learning",
        "What are neural networks?",  # Repeat to increase frequency
        "How does machine learning work?",
        "Describe transformers in NLP",
        "What is reinforcement learning?",
        "Explain neural networks again",  # Another repeat
    ]

    search_count = 0
    for query in queries:
        try:
            logger.info(f"Searching: '{query}'")
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                save_interaction=True,  # Critical: saves interaction to graph
                top_k=5
            )
            search_count += 1
            logger.debug(f"Search completed, got {len(results) if results else 0} results")
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")

    logger.info(f"Completed {search_count} searches with interactions saved")
    return search_count


async def retrieve_interaction_graph() -> List[CogneeGraph]:
    """Retrieve the graph containing interaction nodes."""
    logger.info("Retrieving graph with interaction data...")
    
    graph_engine = await get_graph_engine()
    graph = CogneeGraph()
    
    # Project the full graph including CogneeUserInteraction nodes
    await graph.project_graph_from_db(
        adapter=graph_engine,
        node_properties_to_project=["type", "node_type", "timestamp", "created_at", "text", "name"],
        edge_properties_to_project=["relationship_type", "timestamp", "created_at"],
        directed=True,
    )
    
    logger.info(f"Retrieved graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Count interaction nodes for verification
    interaction_count = sum(
        1 for node in graph.nodes.values()
        if node.attributes.get('type') == 'CogneeUserInteraction' or 
           node.attributes.get('node_type') == 'CogneeUserInteraction'
    )
    logger.info(f"Found {interaction_count} CogneeUserInteraction nodes in graph")
    
    return [graph]


async def run_frequency_pipeline_method1():
    """Method 1: Using the pipeline creation function."""
    logger.info("\n=== Method 1: Using create_usage_frequency_pipeline ===")
    
    graph_engine = await get_graph_engine()
    subgraphs = await retrieve_interaction_graph()
    
    # Create the pipeline tasks
    extraction_tasks, enrichment_tasks = await create_usage_frequency_pipeline(
        graph_adapter=graph_engine,
        time_window=timedelta(days=30),  # Last 30 days
        min_interaction_threshold=1,     # Count all interactions
        batch_size=100
    )
    
    logger.info("Running extraction tasks...")
    # Note: In real memify pipeline, these would be executed by the pipeline runner
    # For this example, we'll execute them manually
    for task in extraction_tasks:
        if hasattr(task, 'function'):
            result = await task.function(
                subgraphs=subgraphs,
                time_window=timedelta(days=30),
                min_interaction_threshold=1
            )
            logger.info(f"Extraction result: {result.get('interactions_in_window')} interactions processed")
    
    logger.info("Running enrichment tasks...")
    for task in enrichment_tasks:
        if hasattr(task, 'function'):
            await task.function(
                graph_adapter=graph_engine,
                usage_frequencies=result
            )
    
    return result


async def run_frequency_pipeline_method2():
    """Method 2: Using the convenience function."""
    logger.info("\n=== Method 2: Using run_usage_frequency_update ===")
    
    graph_engine = await get_graph_engine()
    subgraphs = await retrieve_interaction_graph()
    
    # Run the complete pipeline in one call
    stats = await run_usage_frequency_update(
        graph_adapter=graph_engine,
        subgraphs=subgraphs,
        time_window=timedelta(days=30),
        min_interaction_threshold=1
    )
    
    logger.info("Frequency update statistics:")
    logger.info(f"  Total interactions: {stats['total_interactions']}")
    logger.info(f"  Interactions in window: {stats['interactions_in_window']}")
    logger.info(f"  Nodes with frequency weights: {len(stats['node_frequencies'])}")
    logger.info(f"  Element types: {stats.get('element_type_frequencies', {})}")
    
    return stats


async def analyze_frequency_weights():
    """Analyze and display the frequency weights that were added."""
    logger.info("\n=== Analyzing Frequency Weights ===")
    
    graph_engine = await get_graph_engine()
    graph = CogneeGraph()
    
    # Project graph with frequency weights
    await graph.project_graph_from_db(
        adapter=graph_engine,
        node_properties_to_project=[
            "type", 
            "node_type", 
            "text", 
            "name",
            "frequency_weight",  # Our added property
            "frequency_updated_at"
        ],
        edge_properties_to_project=["relationship_type"],
        directed=True,
    )
    
    # Find nodes with frequency weights
    weighted_nodes = []
    for node_id, node in graph.nodes.items():
        freq_weight = node.attributes.get('frequency_weight')
        if freq_weight is not None:
            weighted_nodes.append({
                'id': node_id,
                'type': node.attributes.get('type') or node.attributes.get('node_type'),
                'text': node.attributes.get('text', '')[:100],  # First 100 chars
                'name': node.attributes.get('name', ''),
                'frequency_weight': freq_weight,
                'updated_at': node.attributes.get('frequency_updated_at')
            })
    
    # Sort by frequency (descending)
    weighted_nodes.sort(key=lambda x: x['frequency_weight'], reverse=True)
    
    logger.info(f"\nFound {len(weighted_nodes)} nodes with frequency weights:")
    logger.info("\nTop 10 Most Frequently Referenced Elements:")
    logger.info("-" * 80)
    
    for i, node in enumerate(weighted_nodes[:10], 1):
        logger.info(f"\n{i}. Frequency: {node['frequency_weight']}")
        logger.info(f"   Type: {node['type']}")
        logger.info(f"   Name: {node['name']}")
        logger.info(f"   Text: {node['text']}")
        logger.info(f"   ID: {node['id'][:50]}...")
    
    return weighted_nodes


async def demonstrate_retrieval_with_frequencies():
    """Demonstrate how frequency weights can be used in retrieval."""
    logger.info("\n=== Demonstrating Retrieval with Frequency Weights ===")
    
    # This is a conceptual demonstration of how frequency weights
    # could be used to boost search results
    
    query = "neural networks"
    logger.info(f"Searching for: '{query}'")
    
    try:
        # Standard search
        standard_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text=query,
            save_interaction=False,  # Don't add more interactions
            top_k=5
        )
        
        logger.info(f"Standard search returned {len(standard_results) if standard_results else 0} results")
        
        # Note: To actually use frequency_weight in scoring, you would need to:
        # 1. Modify the retrieval/ranking logic to consider frequency_weight
        # 2. Add frequency_weight as a scoring factor in the completion strategy
        # 3. Use it in analytics dashboards to show popular topics
        
        logger.info("\nFrequency weights can now be used for:")
        logger.info("  - Boosting frequently-accessed nodes in search rankings")
        logger.info("  - Adjusting triplet importance scores")
        logger.info("  - Building usage analytics dashboards")
        logger.info("  - Identifying 'hot' topics in the knowledge graph")
        
    except Exception as e:
        logger.warning(f"Demonstration search failed: {e}")


async def main():
    """Main execution flow."""
    logger.info("=" * 80)
    logger.info("Usage Frequency Tracking Example")
    logger.info("=" * 80)
    
    try:
        # Step 1: Setup knowledge base
        await setup_knowledge_base()
        
        # Step 2: Simulate user searches with save_interaction=True
        search_count = await simulate_user_searches()
        
        if search_count == 0:
            logger.warning("No searches completed - cannot demonstrate frequency tracking")
            return
        
        # Step 3: Run frequency extraction and enrichment
        # You can use either method - both accomplish the same thing
        
        # Option A: Using the convenience function (recommended)
        stats = await run_frequency_pipeline_method2()
        
        # Option B: Using the pipeline creation function (for custom pipelines)
        # stats = await run_frequency_pipeline_method1()
        
        # Step 4: Analyze the results
        weighted_nodes = await analyze_frequency_weights()
        
        # Step 5: Demonstrate retrieval usage
        await demonstrate_retrieval_with_frequencies()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Searches performed: {search_count}")
        logger.info(f"Interactions tracked: {stats.get('interactions_in_window', 0)}")
        logger.info(f"Nodes weighted: {len(weighted_nodes)}")
        logger.info(f"Time window: {stats.get('time_window_days', 0)} days")
        logger.info("\nFrequency weights have been added to the graph!")
        logger.info("These can now be used in retrieval, ranking, and analytics.")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())