from textblob import TextBlob
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

def run_sentiment_analysis(
    prev_question: str,
    prev_answer: str,
    current_question: str,
    user_id: str
) -> dict[str, str | float]:
    """
    Analyze sentiment from combined conversation context.

    Args:
        prev_question (str): Previous question text
        prev_answer (str): Previous answer text
        current_question (str): Current question text
        user_id (str): ID of the user

    Returns:
        dict[str, str | float]: {
            "user_id": str,
            "sentiment": "positive" | "neutral" | "negative",
            "score": float,
            "context": str
        }
    """

    # Validate input
    if not any([prev_question, prev_answer, current_question]):
        logger.warning(f"Empty input for sentiment analysis, user_id={user_id}")
        return {
            "user_id": user_id,
            "sentiment": "neutral",
            "score": 0.0,
            "context": "",
        }

    # Combine text
    combined_text = f"{prev_question} {prev_answer} {current_question}".strip()

    # Perform sentiment analysis safely
    try:
        polarity = TextBlob(combined_text).sentiment.polarity
    except Exception as e:
        logger.error(f"Sentiment analysis failed for user_id={user_id}: {e}")
        raise ValueError(f"Failed to analyze sentiment: {e}") from e

    # Map score to sentiment label
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    result = {
        "user_id": user_id,
        "sentiment": sentiment,
        "score": round(polarity, 3),
        "context": combined_text,
    }

    logger.info(f"Sentiment analysis result for user_id={user_id}: {result}")
    return result


if __name__ == "__main__":
    import asyncio

    async def _demo():
        result = run_sentiment_analysis(
            prev_question="How are you?",
            prev_answer="I'm good, thanks!",
            current_question="What do you think about the project?",
            user_id="user_123"
        )
        print(result)

    asyncio.run(_demo())
