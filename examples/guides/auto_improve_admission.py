"""Decline the automatic improve() that remember() would start, from the application hosting cognee.

One admission check is registered and answers "insufficient_credits", as a host would for a
user whose LLM budget is spent. The session entry is still stored; no improve is started, and
the result carries the reason (improve_skipped) instead of an ImproveResult.

Requires: LLM_API_KEY. The declined improve makes no LLM call, but storing a session entry
embeds it for session recall: one request to the embedding provider, which with the default
setup is authenticated with that key.
Run: uv run python examples/guides/auto_improve_admission.py
"""

import asyncio

import cognee
from cognee.modules.improve import (
    clear_auto_improve_admission,
    register_auto_improve_admission,
)

DATASET = "demo_dataset"
SESSION = "demo_session"


async def out_of_credit(*, user, dataset_id, session_id, **kwargs) -> str | None:
    # Return None to let the improve run, or a short machine-readable reason to skip it.
    # A real host decides from its own state here, e.g. the balance of the user's account.
    return "insufficient_credits"


async def main():
    # Start clean (optional in your app)
    await cognee.forget(everything=True)

    # One check per process; remember() awaits it before it would start an improve.
    register_auto_improve_admission(out_of_credit)

    result = await cognee.remember(
        "Niels Bohr worked on atomic structure.",
        dataset_name=DATASET,
        session_id=SESSION,
    )

    # Remove the check again, so later remember() calls improve as usual.
    clear_auto_improve_admission()

    entries = await cognee.session.get_session(session_id=SESSION)

    print("Status:", result.status)
    print("Session entries:", [entry.answer for entry in entries])
    print("Improve skipped:", result.improve_skipped)
    print("Improve result:", result.improve)


if __name__ == "__main__":
    asyncio.run(main())
