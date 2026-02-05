import os
import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import setup_logging, INFO


async def main():
    if os.environ.get("CACHING") is None:
        os.environ["CACHING"] = "true"
    if os.environ.get("CACHE_BACKEND") is None:
        os.environ["CACHE_BACKEND"] = "redis"

    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Done.\n")

    texts = [
        "Cognee builds knowledge graphs from text and provides session-based feedback APIs. "
        "You can attach feedback (rating and comment) to each Q&A and later retract it.",
        "Sessions group Q&A by conversation. Use a session_id in search() to keep turns in one thread; "
        "omit it to use the default_session.",
        "Feedback helps improve answers: add_feedback stores a score and optional text, "
        "delete_feedback clears it.",
    ]
    await cognee.add(texts)
    await cognee.cognify()

    user = await get_default_user()

    # ---- Named session: a few questions in one conversation ----
    print("--- Session: product_questions ---")
    session_id = "product_questions"

    for q in [
        "What does Cognee provide?",
        "How do sessions work?",
        "Can I attach feedback to answers?",
    ]:
        print(f"  Q: {q}")
        results = await cognee.search(
            query_text=q,
            query_type=SearchType.GRAPH_COMPLETION,
            user=user,
            session_id=session_id,
        )
        print(f"  A: {results[0] if results else '(no answer)'}\n")

    # Inspect full history for this session
    all_qas = await cognee.session.get_session(session_id=session_id, user=user)
    print(f"  get_session({session_id!r}) → {len(all_qas)} Q&A(s)\n")

    # Show only the last 2 interactions
    recent = await cognee.session.get_session(session_id=session_id, user=user, last_n=2)
    print("  Last 2 turns (last_n=2):")
    for i, e in enumerate(recent, 1):
        print(f"    {i}. Q: {e.question[:50]}... → A: {e.answer[:40] if e.answer else ''}...")
    print()

    # Add feedback to the latest answer (5 stars, helpful)
    latest = all_qas[-1]
    ok = await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=latest.qa_id,
        feedback_text="Very helpful, thanks!",
        feedback_score=5,
        user=user,
    )
    print(f"  add_feedback(latest, 5 stars) → {ok}\n")

    # ---- Default session: one question without a custom session_id ----
    print("--- Session: default_session (no session_id in search) ---")
    results_default = await cognee.search(
        query_text="How are sessions related to Cognee?",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
    )
    print(f"  Q: How are sessions related to Cognee?")
    print(f"  A: {results_default[0] if results_default else '(no answer)'}\n")

    default_qas = await cognee.session.get_session(session_id="default_session", user=user)
    print(f"  get_session('default_session') → {len(default_qas)} Q&A(s)")
    latest_default = default_qas[-1]
    await cognee.session.add_feedback(
        session_id="default_session",
        qa_id=latest_default.qa_id,
        feedback_text="Could be clearer.",
        feedback_score=2,
        user=user,
    )
    print("  add_feedback(latest, 2 stars)\n")

    # ---- Retract feedback (delete_feedback) ----
    print("--- Retract feedback in product_questions ---")
    # Confirm the entry has feedback
    after_add = await cognee.session.get_session(session_id=session_id, user=user)
    entry = next(e for e in after_add if e.qa_id == latest.qa_id)
    print(f"  Before retract: feedback_text={entry.feedback_text!r}, score={entry.feedback_score}")

    deleted = await cognee.session.delete_feedback(
        session_id=session_id, qa_id=latest.qa_id, user=user
    )
    print(f"  delete_feedback(...) → {deleted}")

    after_del = await cognee.session.get_session(session_id=session_id, user=user)
    entry_after = next(e for e in after_del if e.qa_id == latest.qa_id)
    print(
        f"  After retract:  feedback_text={entry_after.feedback_text!r}, score={entry_after.feedback_score}\n"
    )

    print("Done. Session API: get_session (full / last_n), add_feedback, delete_feedback.")


if __name__ == "__main__":
    setup_logging(log_level=INFO)
    asyncio.run(main())
