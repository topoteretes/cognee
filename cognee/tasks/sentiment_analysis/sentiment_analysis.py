# tasks/run_sentiment_analysis.py

from cognee.infrastructure.llm import LLMGateway
from cognee.tasks.storage import add_data_points
from cognee.modules.engine.models import NodeSet
from uuid import uuid5, NAMESPACE_OID
from typing import Optional
from cognee.modules.retrieval.utils.models import CogneeSearchSentiment
from cognee.modules.users.models import User

async def run_sentiment_analysis(prev_question: str, prev_answer: str, current_question: str, user: User):
    text_input = f"""
    Previous Q: {prev_question}
    Answer: {prev_answer}
    Current Q: {current_question}
    """
    user_id = str(user.id)
    # Call LLM to classify sentiment
    sentiment_result = await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt="""Classify the user's reaction as Positive, Neutral, or Negative with a score (-5 to 5).Return the result as valid JSON like:{"sentiment": "Positive","score": 3}""",
        response_model= CogneeSearchSentiment  
    )
    sentiment_data_point = CogneeSearchSentiment(
        id=uuid5(NAMESPACE_OID, name=user_id + current_question),
        prev_question=prev_question,
        prev_answer=prev_answer,
        current_question=current_question,
        sentiment=sentiment_result.sentiment,
        score=sentiment_result.score,
        user_id=user_id,
        belongs_to_set=NodeSet(id=uuid5(NAMESPACE_OID, "CogneeSearchSentiment"), name="CogneeSearchSentiment")
    )
    await add_data_points(data_points=[sentiment_data_point], update_edge_collection=True)
    return {
        "prev_question" : prev_question,
        "prev_answer" : prev_answer,
        "current_question" :current_question,
        "sentiment": sentiment_result.sentiment,
        "score": sentiment_result.score
    }
