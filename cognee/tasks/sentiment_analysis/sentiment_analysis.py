"""
Sentiment Classification Task
-----------------------------
This module performs simple sentiment analysis on text input
and returns a structured sentiment output.
"""

from textblob import TextBlob
import asyncio


async def run_sentiment_analysis(prev_question: str, prev_answer: str, current_question: str, user_id: str):
    """
    Analyze sentiment based on the latest user interaction.
    
    Args:
        prev_question (str): The previous question asked by the user.
        prev_answer (str): The system's previous answer.
        current_question (str): The current user input/question.
        user_id (str): Identifier of the current user.
        
    Returns:
        dict: {
            "user_id": str,
            "sentiment": "positive" | "neutral" | "negative",
            "score": float,
            "context": str
        }
    """
    # Combine conversation context
    combined_text = f"{prev_question} {prev_answer} {current_question}".strip()

    # Perform sentiment analysis using TextBlob
    polarity = TextBlob(combined_text).sentiment.polarity

    if polarity > 0:
        sentiment = "positive"
    elif polarity == 0:
        sentiment = "neutral"
    else:
        sentiment = "negative"

    result = {
        "user_id": user_id,
        "sentiment": sentiment,
        "score": round(polarity, 3),
        "context": combined_text,
    }

    return result


# For quick testing without importing
if __name__ == "__main__":
    async def _demo():
        res = await run_sentiment_analysis(
            prev_question="How are you?",
            prev_answer="I'm doing great!",
            current_question="I love this tool!",
            user_id="demo_user",
        )
        print(res)

    asyncio.run(_demo())
