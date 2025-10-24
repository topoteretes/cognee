from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.models import NodeSet
from uuid import uuid5, NAMESPACE_OID
from typing import Optional, List
from cognee.modules.retrieval.utils.models import CogneeSearchSentiment
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.infrastructure.databases.graph import get_graph_engine

logger = get_logger()

async def run_sentiment_analysis(user: Optional[User] = None) -> List[CogneeSearchSentiment]:
    """
    This function fetches all the nodes from the graph, filters for nodes with a 'question' attribute,
    and performs sentiment analysis on each question.
    Returns a list of sentiment data points for non-neutral sentiments.
    """

    # Fetch all graph data (nodes and edges)
    graph_engine = await get_graph_engine()
    nodes_data, edges_data = await graph_engine.get_graph_data()

    # Filter the nodes to find those with a 'question' attribute
    question_nodes = [node for node in nodes_data if 'question' in node[1]]

    if user is None:
        user = await get_default_user()  # Get default user if no user is provided
    
    user_id = str(user.id)
    
    # Initialize an empty list to store data points
    sentiment_data_points = []

    # For each filtered node (which has a 'question'), perform sentiment analysis
    for node in question_nodes:
        current_question = node[1].get('question')  # Get the question from the node's properties
        if current_question:
            # Call LLM to classify sentiment for the current question
            sentiment_result = await LLMGateway.acreate_structured_output(
                text_input=current_question,
                system_prompt="""Classify the user's reaction as Positive, Neutral, or Negative with a score (-5 to 5).Return the result as valid JSON like:{"sentiment": "Positive","score": 3}""",
                response_model=CogneeSearchSentiment
            )

            # Print sentiment result for debugging
            print(sentiment_result)

            sentiment_data_point = CogneeSearchSentiment(
                id=uuid5(NAMESPACE_OID, name=user_id + current_question),
                current_question=current_question,
                sentiment=sentiment_result.sentiment,
                score=sentiment_result.score,
                user_id=user_id,
                belongs_to_set=NodeSet(id=uuid5(NAMESPACE_OID, "CogneeSearchSentiment"), name="CogneeSearchSentiment")
            )

            # Only add the data point to the list if sentiment is non-neutral
            if sentiment_result.sentiment != 'Neutral':
                sentiment_data_points.append(sentiment_data_point)

                # Log the created data point for debugging
                data_point = {
                    "current_question": current_question,
                    "sentiment": sentiment_result.sentiment,
                    "score": sentiment_result.score
                }
                logger.info(f"Sentiment Data Point Created: {data_point}")

    # Return the list of sentiment data points
    return sentiment_data_points
