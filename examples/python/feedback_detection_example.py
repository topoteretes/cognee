

import asyncio

from cognee.infrastructure.session.feedback_detection import detect_feedback


async def main():
    samples = [
        "What is the capital of France?",
        "thanks, that was helpful!",
        "that was wrong",
        "5/5",
        "can you elaborate on the second point?",
        "nope",
        "Perfect, exactly what I needed.",
    ]

    print("Feedback detection examples\n" + "-" * 50)
    for msg in samples:
        result = await detect_feedback(msg)
        print(f"  Input: {msg!r}")
        print(
            f"  -> feedback_detected={result.feedback_detected}, "
            f"feedback_text={result.feedback_text!r}, "
            f"feedback_score={result.feedback_score}"
        )
        print()
    print("-" * 50 + "\nDone.")




if __name__ == "__main__":
    asyncio.run(main())
