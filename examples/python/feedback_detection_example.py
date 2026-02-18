"""
Example: test the automatic feedback detection across different cases.

Run from repo root with LLM configured (e.g. LLM_API_KEY in .env):

  uv run python examples/python/feedback_detection_example.py

Only messages that validate the correctness/quality of the previous answer
count as feedback. Covers: new questions, follow-ups, topic reactions (not
feedback), praise/criticism/ratings/short reactive (feedback).
"""

import asyncio

from cognee.infrastructure.session.feedback_detection import detect_feedback

# (case label, user message)
SAMPLES = [
    # Not feedback: new questions
    ("New question", "What is the capital of France?"),
    ("New question", "How does Cognee build the knowledge graph?"),
    ("New question", "Explain the difference between add and cognify."),
    # Not feedback: follow-up that asks for more
    ("Follow-up question", "can you elaborate on the second point?"),
    ("Follow-up question", "what about the other approach?"),
    ("Follow-up question", "do you have more details on that?"),
    # Not feedback: topic reaction (does not validate correctness of answer)
    ("Topic reaction (not feedback)", "oooh that place is nice"),
    ("Topic reaction (not feedback)", "interesting, I've been there"),
    ("Topic reaction (not feedback)", "cool story"),
    # Praise / thanks
    ("Praise / thanks", "thanks, that was helpful!"),
    ("Praise / thanks", "Perfect, exactly what I needed."),
    ("Praise / thanks", "that was great, thanks"),
    # Criticism / correction
    ("Criticism / correction", "that was wrong"),
    ("Criticism / correction", "not what I meant, the opposite"),
    ("Criticism / correction", "the date was 2020, not 2021"),
    # Rating / score
    ("Rating", "5/5"),
    ("Rating", "3 stars"),
    ("Rating", "2/5"),
    # Short reactive
    ("Short reactive", "nope"),
    ("Short reactive", "yes"),
    ("Short reactive", "correct"),
]


async def main():
    print("Feedback detection – testing different cases\n" + "=" * 60)
    detected = 0
    for case, msg in SAMPLES:
        result = await detect_feedback(msg)
        if result.feedback_detected:
            detected += 1
        print(f"[{case}]")
        print(f"  Input: {msg!r}")
        print(
            f"  -> detected={result.feedback_detected}, "
            f"score={result.feedback_score}, "
            f"text={result.feedback_text!r}"
        )
        print()
    print("=" * 60)
    print(f"Summary: {detected}/{len(SAMPLES)} messages detected as feedback.\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
