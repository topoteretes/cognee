"""
Demo showing how to use the sentiment analysis memify pipeline.
This demonstrates the complete task: analyzing user interactions for sentiment
as part of the memify pipeline.
"""

import cognee
import asyncio
from cognee.tasks.memify.sentiment_memify_pipeline import sentiment_memify_pipeline, run_sentiment_analysis_only
from cognee.tasks.memify.analyze_interactions_sentiment import get_sentiment_analysis_stats

async def main():
    print("=== Cognee Sentiment Analysis Memify Pipeline Demo ===\n")

    # Create a clean slate for cognee -- reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # Add sample content
    text = "Cognee turns documents into AI memory. It helps users find information quickly and accurately."
    await cognee.add(text)
    
    # Process with LLMs to build the knowledge graph
    print("Building knowledge graph...")
    await cognee.cognify()
    print("Knowledge graph built successfully!\n")
    
    # Perform several searches with save_interaction=True to generate interaction data
    print("Performing searches with interaction saving enabled...")
    
    search_queries = [
        "What does Cognee do?",
        "How does Cognee help with memory?",
        "Tell me about AI memory systems",
        "What are the benefits of using Cognee?",
        "How accurate is Cognee's search?"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"Search {i}: {query}")
        results = await cognee.search(
            query_text=query,
            save_interaction=True  # This will save the interaction for sentiment analysis
        )
        print(f"Results: {len(results)} found")
        for result in results:
            print(f"  - {result}")
        print()
    
    # Now let's test the memify pipeline with sentiment analysis
    print("=== Testing Memify Pipeline with Sentiment Analysis ===\n")
    
    # Run the sentiment analysis memify pipeline
    print("Running memify pipeline with sentiment analysis...")
    pipeline_result = await sentiment_memify_pipeline(
        include_sentiment_analysis=True,
        sentiment_batch_size=5,
        sentiment_limit=10
    )
    
    print("Memify pipeline completed!")
    print(f"Sentiment Analysis Results:")
    
    sentiment_data = pipeline_result.get("sentiment_analysis", {})
    if "error" in sentiment_data:
        print(f"Error: {sentiment_data['error']}")
    else:
        analysis_result = sentiment_data.get("sentiment_analysis", {})
        print(f"  - Interactions Processed: {analysis_result.get('processed', 0)}")
        print(f"  - Total Interactions Found: {analysis_result.get('summary', {}).get('total_interactions', 0)}")
        print(f"  - Processing Percentage: {analysis_result.get('summary', {}).get('processing_percentage', 0)}%")
    
    # Get overall sentiment statistics
    print("\n=== Getting Sentiment Statistics ===")
    stats = await get_sentiment_analysis_stats()
    
    if "error" not in stats:
        print(f"Overall Sentiment Statistics:")
        print(f"  - Total Interactions: {stats.get('total_interactions', 0)}")
        print(f"  - Positive: {stats.get('positive', 0)}")
        print(f"  - Negative: {stats.get('negative', 0)}")
        print(f"  - Neutral: {stats.get('neutral', 0)}")
        print(f"  - Average Score: {stats.get('average_score', 0)}")
        print(f"  - Sentiment Distribution:")
        dist = stats.get('sentiment_distribution', {})
        print(f"    * Positive: {dist.get('positive_pct', 0)}%")
        print(f"    * Negative: {dist.get('negative_pct', 0)}%")
        print(f"    * Neutral: {dist.get('neutral_pct', 0)}%")
    else:
        print(f"Error getting statistics: {stats['error']}")
    
    # Demonstrate standalone sentiment analysis
    print("\n=== Testing Standalone Sentiment Analysis ===")
    print("Running standalone sentiment analysis on all interactions...")
    
    standalone_result = await run_sentiment_analysis_only(
        limit=20,
        batch_size=5,
        include_processed=True  # Include already processed interactions
    )
    
    standalone_data = standalone_result.get("sentiment_result", {})
    print(f"Standalone Analysis Results:")
    print(f"  - Processed: {standalone_data.get('processed', 0)}")
    print(f"  - Summary: {standalone_data.get('summary', {})}")
    
    print("\n=== Demo Complete ===")
    print("The sentiment analysis is now fully integrated into the memify pipeline!")
    print("This provides insights into how users react to Cognee search results.")

if __name__ == '__main__':
    asyncio.run(main())
