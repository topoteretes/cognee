"""Edge-case grounded run — candidate outside R4's listed companies.

Maria Cruz is ex-Revolut (a fintech, but not in R4's enumerated list of
Stripe/Plaid/Adyen). So `compose_screen_invite` has no clearly triggering
rule — it should either leave `applied_rule_ids` empty (→ `novel` bucket
in the trace linker) or propose a new rule extending R4 to cover Revolut
(→ `novel` bucket + something to review in `review_pending_rules.py`).

Uses a dedicated session_id so it doesn't collide with the main Dev Rao
traces from run_grounded.py — run both linkers separately.
"""

import asyncio
import os

os.environ.setdefault("RECRUITING_WITH_MEMORY", "true")
os.environ.setdefault("RECRUITING_SESSION_ID", "recruiting-demo-grounded-maria")
os.environ.setdefault("RECRUITING_CANDIDATE", "maria_cruz")

from examples.demos.recruiting_distill_memory._run import run_plan  # noqa: E402


if __name__ == "__main__":
    asyncio.run(run_plan("grounded_plan_maria.json"))
