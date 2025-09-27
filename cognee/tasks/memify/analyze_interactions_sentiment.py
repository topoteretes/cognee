"""
Memify task for analyzing sentiment of saved interactions.
This task processes interactions that were saved when save_interaction=True was used
and performs sentiment analysis on them as part of the memify pipeline.
"""

from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_OID, uuid5
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.user_qa_feedback import UserQAFeedback
from cognee.modules.engine.models import NodeSet
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.engine.models.DataPoint import DataPoint

logger = get_logger("analyze_interactions_sentiment")


class InteractionSentimentDataPoint(DataPoint):
    """Data point for storing interaction sentiment analysis results"""
    interaction_id: str
    question: str
    answer: str
    sentiment: str
    score: float
    belongs_to_set: Optional[NodeSet] = None


async def analyze_interactions_sentiment(input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Analyze sentiment of saved interactions as part of the memify pipeline.
    
    This task processes interactions that were saved when save_interaction=True was used
    and performs sentiment analysis on them to understand user satisfaction.
    
    Args:
        input_data: Dictionary containing:
            - 'limit': Optional limit on number of interactions to analyze (default: 100)
            - 'batch_size': Optional batch size for processing (default: 10)
            - 'include_processed': Whether to include already processed interactions (default: False)
    
    Returns:
        Dictionary with analysis results and statistics
    """
    limit = input_data.get("limit", 100)
    batch_size = input_data.get("batch_size", 10)
    include_processed = input_data.get("include_processed", False)
    
    logger.info(f"Starting sentiment analysis for interactions (limit: {limit})")
    
    try:
        interactions = await _get_saved_interactions(limit, include_processed)
        
        if not interactions:
            logger.info("No interactions found for sentiment analysis")
            return {"processed": 0, "sentiments": []}
        
        nodeset_name = "InteractionSentiments"
        sentiment_node_set = NodeSet(
            id=uuid5(NAMESPACE_OID, name=nodeset_name), 
            name=nodeset_name
        )
        
        processed_count = 0
        sentiment_results = []
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            
            for interaction in batch:
                try:
                    interaction_id = interaction.get("id", "")
                    question = interaction.get("question", "")
                    answer = interaction.get("answer", "")
                    
                    feedback_text = f"Question: {question}\nAnswer: {answer}"
                    feedback_analyzer = UserQAFeedback(last_k=1)
                    await feedback_analyzer.add_feedback(feedback_text)
                    
                    sentiment_id = uuid5(NAMESPACE_OID, f"sentiment_{interaction_id}")
                    sentiment_datapoint = InteractionSentimentDataPoint(
                        id=sentiment_id,
                        interaction_id=interaction_id,
                        question=question,
                        answer=answer,
                        sentiment="analyzed", 
                        score=0.0, 
                        belongs_to_set=sentiment_node_set
                    )
                    
                    await add_data_points(
                        data_points=[sentiment_datapoint], 
                        update_edge_collection=False
                    )
                    
                    await _create_sentiment_relationship(interaction_id, sentiment_id)
                    
                    sentiment_results.append({
                        "interaction_id": interaction_id,
                        "question": question,
                        "answer": answer[:100] + "..." if len(answer) > 100 else answer
                    })
                    
                    processed_count += 1
                    logger.info(f"Processed interaction {interaction_id} for sentiment analysis")
                    
                except Exception as e:
                    logger.error(f"Error processing interaction {interaction.get('id', 'unknown')}: {str(e)}")
                    continue

        summary = {
            "total_processed": processed_count,
            "total_interactions": len(interactions),
            "processing_percentage": round((processed_count / len(interactions)) * 100, 1) if interactions else 0
        }
        
        logger.info(f"Sentiment analysis complete: {summary}")
        
        return {
            "processed": processed_count,
            "sentiments": sentiment_results,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis task: {str(e)}")
        return {"error": str(e), "processed": 0}


async def _get_saved_interactions(limit: int, include_processed: bool = False) -> List[Dict[str, Any]]:
    """Get saved interactions from the graph database"""
    try:
        graph_engine = await get_graph_engine()
        
        if include_processed:
          
            query = """
            MATCH (n:InteractionDataPoint)
            WHERE n.belongs_to_set.name = 'Interactions'
            RETURN n.id as id, n.question as question, n.answer as answer, n.context as context
            ORDER BY n.created_at DESC
            LIMIT $limit
            """
        else:
           
            query = """
            MATCH (n:InteractionDataPoint)
            WHERE n.belongs_to_set.name = 'Interactions'
            AND NOT EXISTS {
                MATCH (n)-[:gives_feedback_to]->(f:CogneeUserFeedback)
            }
            RETURN n.id as id, n.question as question, n.answer as answer, n.context as context
            ORDER BY n.created_at DESC
            LIMIT $limit
            """
        
        results = await graph_engine.execute_query(query, {"limit": limit})
        
        interactions = []
        for result in results:
            interaction = {
                "id": str(result["id"]),
                "question": result["question"],
                "answer": result["answer"],
                "context": result.get("context", "")
            }
            interactions.append(interaction)
        
        return interactions
        
    except Exception as e:
        logger.error(f"Error getting saved interactions: {str(e)}")
        return []


async def _create_sentiment_relationship(interaction_id: str, sentiment_id: str) -> None:
    """Create a relationship between interaction and its sentiment analysis"""
    try:
        graph_engine = await get_graph_engine()
        
        relationship = (
            uuid5(NAMESPACE_OID, interaction_id),
            sentiment_id,
            "has_sentiment_analysis",
            {
                "relationship_name": "has_sentiment_analysis",
                "source_node_id": uuid5(NAMESPACE_OID, interaction_id),
                "target_node_id": sentiment_id,
                "ontology_valid": False,
            }
        )
        
        await graph_engine.add_edges([relationship])
        
    except Exception as e:
        logger.error(f"Error creating sentiment relationship: {str(e)}")


async def get_sentiment_analysis_stats() -> Dict[str, Any]:
    """Get statistics about sentiment analysis results"""
    try:
        graph_engine = await get_graph_engine()
        
        query = """
        MATCH (n:CogneeUserFeedback)
        WHERE n.belongs_to_set.name = 'UserQAFeedbacks'
        RETURN n.sentiment as sentiment, n.score as score
        """
        
        results = await graph_engine.execute_query(query)
        
        if not results:
            return {"message": "No sentiment data found"}
        
        sentiments = [r["sentiment"] for r in results]
        scores = [r["score"] for r in results]
        
        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")
        neutral_count = sentiments.count("neutral")
        
        return {
            "total_interactions": len(results),
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "average_score": round(sum(scores) / len(scores), 2),
            "sentiment_distribution": {
                "positive_pct": round((positive_count / len(results)) * 100, 1),
                "negative_pct": round((negative_count / len(results)) * 100, 1),
                "neutral_pct": round((neutral_count / len(results)) * 100, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment stats: {str(e)}")
        return {"error": str(e)}
