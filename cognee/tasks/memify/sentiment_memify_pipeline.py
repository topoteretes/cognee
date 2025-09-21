"""
Sentiment Analysis Memify Pipeline

This module provides a complete memify pipeline that includes sentiment analysis
of user interactions. It can be used to analyze how users react to Cognee search
results and provide insights for improving the system.
"""

from typing import List, Optional, Union
from uuid import UUID
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.memify.memify import memify
from cognee.tasks.memify.analyze_interactions_sentiment import analyze_interactions_sentiment
from cognee.modules.engine.models import NodeSet


async def sentiment_memify_pipeline(
    extraction_tasks: Optional[List[Task]] = None,
    enrichment_tasks: Optional[List[Task]] = None,
    data: Optional[any] = None,
    dataset: Union[str, UUID] = "main_dataset",
    user: Optional[any] = None,
    node_type: Optional[type] = NodeSet,
    node_name: Optional[List[str]] = None,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    run_in_background: bool = False,
    include_sentiment_analysis: bool = True,
    sentiment_batch_size: int = 10,
    sentiment_limit: int = 100,
):
    """
    Enhanced memify pipeline that includes sentiment analysis of user interactions.
    
    This pipeline extends the standard memify functionality by automatically
    analyzing user interactions (when save_interaction=True was used) to understand
    user satisfaction and reaction to Cognee search results.
    
    Args:
        extraction_tasks: List of Cognee Tasks to execute for graph/data extraction.
        enrichment_tasks: List of Cognee Tasks to handle enrichment of provided graph/data.
        data: The data to ingest. Can be anything when custom extraction and enrichment tasks are used.
        dataset: Dataset name or dataset uuid to process.
        user: User context for authentication and data access.
        node_type: Filter graph to specific entity types.
        node_name: Filter graph to specific named entities.
        vector_db_config: Custom vector database configuration for embeddings storage.
        graph_db_config: Custom graph database configuration for relationship storage.
        run_in_background: If True, starts processing asynchronously and returns immediately.
        include_sentiment_analysis: Whether to include sentiment analysis in the pipeline.
        sentiment_batch_size: Batch size for sentiment analysis processing.
        sentiment_limit: Maximum number of interactions to analyze for sentiment.
    
    Returns:
        Dictionary containing pipeline results and sentiment analysis statistics.
    """
    
    memify_result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=data,
        dataset=dataset,
        user=user,
        node_type=node_type,
        node_name=node_name,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        run_in_background=run_in_background,
    )
    
    sentiment_results = {}
    
    if include_sentiment_analysis:
        try:
            sentiment_task = Task(analyze_interactions_sentiment)
            sentiment_result = await sentiment_task.func({
                "limit": sentiment_limit,
                "batch_size": sentiment_batch_size,
                "include_processed": False  
            })
            
            sentiment_results = {
                "sentiment_analysis": sentiment_result,
                "interactions_processed": sentiment_result.get("processed", 0)
            }
                
        except Exception as e:
            sentiment_results = {
                "sentiment_analysis": {"error": str(e)},
                "interactions_processed": 0
            }
    
    return {
        "memify_result": memify_result,
        "sentiment_analysis": sentiment_results
    }


async def create_sentiment_analysis_tasks() -> List[Task]:
    """
    Create a list of tasks for sentiment analysis that can be used in memify pipeline.
    
    Returns:
        List of Task objects for sentiment analysis
    """
    return [
        Task(analyze_interactions_sentiment, limit=100, batch_size=10)
    ]


async def run_sentiment_analysis_only(
    limit: int = 100,
    batch_size: int = 10,
    include_processed: bool = False
) -> dict:
    """
    Run only the sentiment analysis part without the full memify pipeline.
    
    This is useful for analyzing existing interactions without running the full
    memify pipeline.
    
    Args:
        limit: Maximum number of interactions to analyze.
        batch_size: Batch size for processing.
        include_processed: Whether to include already processed interactions.
    
    Returns:
        Dictionary containing sentiment analysis results.
    """
    sentiment_task = Task(analyze_interactions_sentiment)
    sentiment_result = await sentiment_task.func({
        "limit": limit,
        "batch_size": batch_size,
        "include_processed": include_processed
    })
    
    return {
        "sentiment_result": sentiment_result
    }
