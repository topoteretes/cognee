import asyncio
import os
import sys

# Let the session capture the user's stated preference as learned guidance.
os.environ["AUTO_FEEDBACK"] = "true"
os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee
from cognee import SearchType
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.methods import get_default_user

SESSION_ID = "snack_session"

# flavor -> (snack that has it, statement of the preference)
SNACK_FOR_FLAVOR = {"savory": "Doritos", "sweet": "Oreos"}


def progress(message: str):
    print(f"[snack-demo] {message}", file=sys.stderr, flush=True)


def answer_text(result) -> str:
    """recall() returns a list of response entries; join their text for parsing/printing."""
    if isinstance(result, str):
        return result
    parts = []
    for entry in result or []:
        parts.append(getattr(entry, "text", None) or str(entry))
    return " ".join(parts)


def recommended_snack(text: str) -> str:
    """Whichever snack the model recommends first in its answer."""
    lowered = text.lower()
    oreo_at = lowered.find("oreo")
    dorito_at = lowered.find("dorito")
    if oreo_at == -1 and dorito_at == -1:
        return "Oreos"  # fallback; shouldn't happen with the snack facts in context
    if dorito_at == -1:
        return "Oreos"
    if oreo_at == -1:
        return "Doritos"
    return "Oreos" if oreo_at < dorito_at else "Doritos"


async def ask(message: str, user, session_id: str):
    # RAG_COMPLETION answers from retrieved chunks. Before distillation only the two snack
    # facts exist, so the model has no basis to prefer one. After distillation the curated
    # preference lesson is a retrievable chunk, so it steers the pick.
    return await cognee.recall(
        query_text=message,
        query_type=SearchType.RAG_COMPLETION,
        datasets=["snack_preference_demo"],
        session_id=session_id,
        user=user,
    )


async def main():
    progress("Clearing previous demo state.")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    progress("Ingesting the two snack facts.")
    await cognee.remember(
        [
            "Oreos are a sweet snack: chocolate cookies with a sugary cream filling.",
            "Doritos are a savory snack: salty, cheesy, seasoned tortilla chips.",
        ],
        dataset_name="snack_preference_demo",
    )

    user = await get_default_user()
    await get_session_manager().delete_session(user_id=str(user.id), session_id=SESSION_ID)

    question = "I want a snack. Should I get Oreos or Doritos? Recommend one."

    # 1) Before distillation: no preference known -> arbitrary pick.
    progress("Asking BEFORE distillation (no preference known).")
    before = answer_text(await ask(question, user, SESSION_ID))
    first_pick = recommended_snack(before)
    print("\n----- BEFORE distillation -----\n", file=sys.stderr)
    print(f"picked: {first_pick}\n{before}", file=sys.stderr)

    # 2) State the OPPOSITE preference so the answer has to flip to the other snack.
    if first_pick == "Doritos":
        preferred_flavor, opposite_flavor = "sweet", "savory"
    else:
        preferred_flavor, opposite_flavor = "savory", "sweet"
    expected_after = SNACK_FOR_FLAVOR[preferred_flavor]
    progress(
        f"Model picked {first_pick}; telling it the user prefers {preferred_flavor} "
        f"(expect it to flip to {expected_after})."
    )
    await ask(
        f"Just so you know, I always prefer {preferred_flavor} snacks over {opposite_flavor} ones.",
        user,
        SESSION_ID,
    )

    # 3) Distill the session into long-term memory.
    progress("Distilling the session into the graph.")
    result = await cognee.session.distill_session(
        SESSION_ID, dataset="snack_preference_demo", user=user
    )
    progress(f"Distillation status={result.status} documents={len(result.documents)}")
    for doc in result.documents:
        print("\n----- distilled lesson -----\n", file=sys.stderr)
        print(doc, file=sys.stderr)

    # 4) After distillation, in a FRESH session, ask the same question again.
    progress("Asking AFTER distillation in a fresh session.")
    after = answer_text(await ask(question, user, "snack_verification_session"))
    second_pick = recommended_snack(after)
    print(f"\n----- AFTER distillation (expected {expected_after}) -----\n", file=sys.stderr)
    print(f"picked: {second_pick}\n{after}", file=sys.stderr)

    flipped = second_pick == expected_after and second_pick != first_pick
    progress(
        f"RESULT: {first_pick} -> {second_pick} "
        f"({'flipped as expected ✅' if flipped else 'did NOT flip ❌'})"
    )


if __name__ == "__main__":
    asyncio.run(main())
