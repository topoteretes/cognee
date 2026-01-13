#!/usr/bin/env python3
"""
End-to-End Example: Usage Frequency Tracking in Cognee

This example demonstrates the complete workflow for tracking and analyzing
how frequently different graph elements are accessed through user searches.

Features demonstrated:
- Setting up a knowledge base
- Running searches with interaction tracking (save_interaction=True)
- Extracting usage frequencies from interaction data
- Applying frequency weights to graph nodes
- Analyzing and visualizing the results

Use cases:
- Ranking search results by popularity
- Identifying "hot topics" in your knowledge base
- Understanding user behavior and interests
- Improving retrieval based on usage patterns
"""

import asyncio
import os
from datetime import timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.tasks.memify.extract_usage_frequency import run_usage_frequency_update

# Load environment variables
load_dotenv()


# ============================================================================
# STEP 1: Setup and Configuration
# ============================================================================

async def setup_knowledge_base():
    """
    Create a fresh knowledge base with sample content.
    
    In a real application, you would:
    - Load documents from files, databases, or APIs
    - Process larger datasets
    - Organize content by datasets/categories
    """
    print("=" * 80)
    print("STEP 1: Setting up knowledge base")
    print("=" * 80)
    
    # Reset state for clean demo (optional in production)
    print("\nResetting Cognee state...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("✓ Reset complete")
    
    # Sample content: AI/ML educational material
    documents = [
        """
        Machine Learning Fundamentals:
        Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        The three main types are supervised learning, unsupervised learning, and
        reinforcement learning.
        """,
        """
        Neural Networks Explained:
        Neural networks are computing systems inspired by biological neural networks.
        They consist of layers of interconnected nodes (neurons) that process information
        through weighted connections. Deep learning uses neural networks with many layers
        to automatically learn hierarchical representations of data.
        """,
        """
        Natural Language Processing:
        NLP enables computers to understand, interpret, and generate human language.
        Modern NLP uses transformer architectures like BERT and GPT, which have
        revolutionized tasks such as translation, summarization, and question answering.
        """,
        """
        Computer Vision Applications:
        Computer vision allows machines to interpret visual information from the world.
        Convolutional neural networks (CNNs) are particularly effective for image
        recognition, object detection, and image segmentation tasks.
        """,
    ]
    
    print(f"\nAdding {len(documents)} documents to knowledge base...")
    await cognee.add(documents, dataset_name="ai_ml_fundamentals")
    print("✓ Documents added")
    
    # Build knowledge graph
    print("\nBuilding knowledge graph (cognify)...")
    await cognee.cognify()
    print("✓ Knowledge graph built")
    
    print("\n" + "=" * 80)


# ============================================================================
# STEP 2: Simulate User Searches with Interaction Tracking
# ============================================================================

async def simulate_user_searches(queries: List[str]):
    """
    Simulate users searching the knowledge base.
    
    The key parameter is save_interaction=True, which creates:
    - CogneeUserInteraction nodes (one per search)
    - used_graph_element_to_answer edges (connecting queries to relevant nodes)
    
    Args:
        queries: List of search queries to simulate
        
    Returns:
        Number of successful searches
    """
    print("=" * 80)
    print("STEP 2: Simulating user searches with interaction tracking")
    print("=" * 80)
    
    successful_searches = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\nSearch {i}/{len(queries)}: '{query}'")
        try:
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                save_interaction=True,  # ← THIS IS CRITICAL!
                top_k=5
            )
            successful_searches += 1
            
            # Show snippet of results
            result_preview = str(results)[:100] if results else "No results"
            print(f"  ✓ Completed ({result_preview}...)")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n✓ Completed {successful_searches}/{len(queries)} searches")
    print("=" * 80)
    
    return successful_searches


# ============================================================================
# STEP 3: Extract and Apply Usage Frequencies
# ============================================================================

async def extract_and_apply_frequencies(
    time_window_days: int = 7,
    min_threshold: int = 1
) -> Dict[str, Any]:
    """
    Extract usage frequencies from interactions and apply them to the graph.
    
    This function:
    1. Retrieves the graph with interaction data
    2. Counts how often each node was accessed
    3. Writes frequency_weight property back to nodes
    
    Args:
        time_window_days: Only count interactions from last N days
        min_threshold: Minimum accesses to track (filter out rarely used nodes)
        
    Returns:
        Dictionary with statistics about the frequency update
    """
    print("=" * 80)
    print("STEP 3: Extracting and applying usage frequencies")
    print("=" * 80)
    
    # Get graph adapter
    graph_engine = await get_graph_engine()
    
    # Retrieve graph with interactions
    print("\nRetrieving graph from database...")
    graph = CogneeGraph()
    await graph.project_graph_from_db(
        adapter=graph_engine,
        node_properties_to_project=[
            "type", "node_type", "timestamp", "created_at",
            "text", "name", "query_text", "frequency_weight"
        ],
        edge_properties_to_project=["relationship_type", "timestamp"],
        directed=True,
    )
    
    print(f"✓ Retrieved: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Count interaction nodes
    interaction_nodes = [
        n for n in graph.nodes.values()
        if n.attributes.get('type') == 'CogneeUserInteraction' or
           n.attributes.get('node_type') == 'CogneeUserInteraction'
    ]
    print(f"✓ Found {len(interaction_nodes)} interaction nodes")
    
    # Run frequency extraction and update
    print(f"\nExtracting frequencies (time window: {time_window_days} days)...")
    stats = await run_usage_frequency_update(
        graph_adapter=graph_engine,
        subgraphs=[graph],
        time_window=timedelta(days=time_window_days),
        min_interaction_threshold=min_threshold
    )
    
    print(f"\n✓ Frequency extraction complete!")
    print(f"  - Interactions processed: {stats['interactions_in_window']}/{stats['total_interactions']}")
    print(f"  - Nodes weighted: {len(stats['node_frequencies'])}")
    print(f"  - Element types tracked: {stats.get('element_type_frequencies', {})}")
    
    print("=" * 80)
    
    return stats


# ============================================================================
# STEP 4: Analyze and Display Results
# ============================================================================

async def analyze_results(stats: Dict[str, Any]):
    """
    Analyze and display the frequency tracking results.
    
    Shows:
    - Top most frequently accessed nodes
    - Element type distribution
    - Verification that weights were written to database
    
    Args:
        stats: Statistics from frequency extraction
    """
    print("=" * 80)
    print("STEP 4: Analyzing usage frequency results")
    print("=" * 80)
    
    # Display top nodes by frequency
    if stats['node_frequencies']:
        print("\n📊 Top 10 Most Frequently Accessed Elements:")
        print("-" * 80)
        
        sorted_nodes = sorted(
            stats['node_frequencies'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get graph to display node details
        graph_engine = await get_graph_engine()
        graph = CogneeGraph()
        await graph.project_graph_from_db(
            adapter=graph_engine,
            node_properties_to_project=["type", "text", "name"],
            edge_properties_to_project=[],
            directed=True,
        )
        
        for i, (node_id, frequency) in enumerate(sorted_nodes[:10], 1):
            node = graph.get_node(node_id)
            if node:
                node_type = node.attributes.get('type', 'Unknown')
                text = node.attributes.get('text') or node.attributes.get('name') or ''
                text_preview = text[:60] + "..." if len(text) > 60 else text
                
                print(f"\n{i}. Frequency: {frequency} accesses")
                print(f"   Type: {node_type}")
                print(f"   Content: {text_preview}")
            else:
                print(f"\n{i}. Frequency: {frequency} accesses")
                print(f"   Node ID: {node_id[:50]}...")
    
    # Display element type distribution
    if stats.get('element_type_frequencies'):
        print("\n\n📈 Element Type Distribution:")
        print("-" * 80)
        type_dist = stats['element_type_frequencies']
        for elem_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {elem_type}: {count} accesses")
    
    # Verify weights in database (Neo4j only)
    print("\n\n🔍 Verifying weights in database...")
    print("-" * 80)
    
    graph_engine = await get_graph_engine()
    adapter_type = type(graph_engine).__name__
    
    if adapter_type == 'Neo4jAdapter':
        try:
            result = await graph_engine.query("""
                MATCH (n)
                WHERE n.frequency_weight IS NOT NULL
                RETURN count(n) as weighted_count
            """)
            
            count = result[0]['weighted_count'] if result else 0
            if count > 0:
                print(f"✓ {count} nodes have frequency_weight in Neo4j database")
                
                # Show sample
                sample = await graph_engine.query("""
                    MATCH (n)
                    WHERE n.frequency_weight IS NOT NULL
                    RETURN n.frequency_weight as weight, labels(n) as labels
                    ORDER BY n.frequency_weight DESC
                    LIMIT 3
                """)
                
                print("\nSample weighted nodes:")
                for row in sample:
                    print(f"  - Weight: {row['weight']}, Type: {row['labels']}")
            else:
                print("⚠ No nodes with frequency_weight found in database")
        except Exception as e:
            print(f"Could not verify in Neo4j: {e}")
    else:
        print(f"Database verification not implemented for {adapter_type}")
    
    print("\n" + "=" * 80)


# ============================================================================
# STEP 5: Demonstrate Usage in Retrieval
# ============================================================================

async def demonstrate_retrieval_usage():
    """
    Demonstrate how frequency weights can be used in retrieval.
    
    Note: This is a conceptual demonstration. To actually use frequency
    weights in ranking, you would need to modify the retrieval/completion
    strategies to incorporate the frequency_weight property.
    """
    print("=" * 80)
    print("STEP 5: How to use frequency weights in retrieval")
    print("=" * 80)
    
    print("""
    Frequency weights can be used to improve search results:
    
    1. RANKING BOOST:
       - Multiply relevance scores by frequency_weight
       - Prioritize frequently accessed nodes in results
       
    2. COMPLETION STRATEGIES:
       - Adjust triplet importance based on usage
       - Filter out rarely accessed information
       
    3. ANALYTICS:
       - Track trending topics over time
       - Understand user interests and behavior
       - Identify knowledge gaps (low-frequency nodes)
       
    4. ADAPTIVE RETRIEVAL:
       - Personalize results based on team usage patterns
       - Surface popular answers faster
       
    Example Cypher query with frequency boost (Neo4j):
    
        MATCH (n)
        WHERE n.text CONTAINS $search_term
        RETURN n, n.frequency_weight as boost
        ORDER BY (n.relevance_score * COALESCE(n.frequency_weight, 1)) DESC
        LIMIT 10
    
    To integrate this into Cognee, you would modify the completion
    strategy to include frequency_weight in the scoring function.
    """)
    
    print("=" * 80)


# ============================================================================
# MAIN: Run Complete Example
# ============================================================================

async def main():
    """
    Run the complete end-to-end usage frequency tracking example.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Usage Frequency Tracking - End-to-End Example".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Configuration check
    print("Configuration:")
    print(f"  Graph Provider: {os.getenv('GRAPH_DATABASE_PROVIDER')}")
    print(f"  Graph Handler: {os.getenv('GRAPH_DATASET_HANDLER')}")
    print(f"  LLM Provider: {os.getenv('LLM_PROVIDER')}")
    
    # Verify LLM key is set
    if not os.getenv('LLM_API_KEY') or os.getenv('LLM_API_KEY') == 'sk-your-key-here':
        print("\n⚠ WARNING: LLM_API_KEY not set in .env file")
        print("   Set your API key to run searches")
        return
    
    print("\n")
    
    try:
        # Step 1: Setup
        await setup_knowledge_base()
        
        # Step 2: Simulate searches
        # Note: Repeat queries increase frequency for those topics
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?",
            "Tell me about neural networks",  # Repeat - increases frequency
            "What are transformers in NLP?",
            "Explain neural networks again",  # Another repeat
            "How does computer vision work?",
            "What is reinforcement learning?",
            "Tell me more about neural networks",  # Third repeat
        ]
        
        successful_searches = await simulate_user_searches(queries)
        
        if successful_searches == 0:
            print("⚠ No searches completed - cannot demonstrate frequency tracking")
            return
        
        # Step 3: Extract frequencies
        stats = await extract_and_apply_frequencies(
            time_window_days=7,
            min_threshold=1
        )
        
        # Step 4: Analyze results
        await analyze_results(stats)
        
        # Step 5: Show usage examples
        await demonstrate_retrieval_usage()
        
        # Summary
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  Example Complete!".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")
        
        print("Summary:")
        print(f"  ✓ Documents added: 4")
        print(f"  ✓ Searches performed: {successful_searches}")
        print(f"  ✓ Interactions tracked: {stats['interactions_in_window']}")
        print(f"  ✓ Nodes weighted: {len(stats['node_frequencies'])}")
        
        print("\nNext steps:")
        print("  1. Open Neo4j Browser (http://localhost:7474) to explore the graph")
        print("  2. Modify retrieval strategies to use frequency_weight")
        print("  3. Build analytics dashboards using element_type_frequencies")
        print("  4. Run periodic frequency updates to track trends over time")
        
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())