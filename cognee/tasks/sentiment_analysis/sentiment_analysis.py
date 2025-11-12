from typing import Optional
from textblob import TextBlob
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

def run_sentiment_analysis(
    prev_question: Optional[str],
    prev_answer: Optional[str],
    current_question: Optional[str],
    user_id: str
) -> dict[str, str | float]:
    """
    Analyzes the sentiment polarity of a conversation by combining previous question, 
    previous answer, and current question, and returns a sentiment label with score.

    Args:
        prev_question (str | None): Previous user question
        prev_answer (str | None): Previous system answer
        current_question (str | None): Current user question
        user_id (str): Unique user ID

    Returns:
        dict: { "user_id": ..., "sentiment": ..., "score": ..., "context": ... }

    Example:
        res = run_sentiment_analysis("Hello", "Hi!", "What's up?", "user_123")
        print(res)
    """
    # Normalize None to empty strings
    prev_question = prev_question or ""
    prev_answer = prev_answer or ""
    current_question = current_question or ""

    if not any([prev_question, prev_answer, current_question]):
        logger.warning(f"Empty input for sentiment analysis, user_id={user_id}")
        return {
            "user_id": user_id,
            "sentiment": "neutral",
            "score": 0.0,
            "context": "",
        }

    # Combine context
    combined_text = f"{prev_question} {prev_answer} {current_question}".strip()

    try:
        blob = TextBlob(combined_text)
        polarity = float(blob.sentiment.polarity)
        # Thresholds: polarity > 0.1 is positive, < -0.1 is negative, else neutral
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
            "context": combined_text,  # Note: Be careful with PII in production!
        }
        logger.debug(
            f"Sentiment analysis completed: user_id={user_id}, sentiment={sentiment}, score={round(polarity,3)}"
        )
        return result
    except Exception as e:
        logger.error(
            f"Sentiment analysis failed: user_id={user_id}, error={e}", exc_info=True
        )
        raise ValueError("Sentiment analysis processing failed") from e

if __name__ == "__main__":
    result = run_sentiment_analysis(
        prev_question="How are you?",
        prev_answer="I'm good, thanks!",
        current_question="What do you think about the project?",
        user_id="user_123"
    )
    print(result)
