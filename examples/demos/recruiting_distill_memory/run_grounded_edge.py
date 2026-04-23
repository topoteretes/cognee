"""Edge-case grounded run — candidate outside R4's listed companies.

Maria Cruz is ex-Revolut (a fintech, but not in R4's enumerated list of
Stripe/Plaid/Adyen). The tools have no rule that unambiguously fires on
her profile, so the decorator's session-feedback summaries will describe
what the tools actually did (e.g. "drafted a screening invite for an
ex-Revolut candidate"). The post-loop memify pipeline then cognifies
those summaries into `agent_proposed_rule` nodes, which a human reviews
via `review_pending_rules.py` — the canonical "agent proposes, human
approves" path.

Uses a dedicated session_id so the Maria traces don't mix with the main
Dev Rao traces from run_grounded.py.
"""

import asyncio
import os

os.environ.setdefault("RECRUITING_WITH_MEMORY", "true")
os.environ.setdefault("RECRUITING_SESSION_ID", "recruiting-demo-grounded-maria")
os.environ.setdefault("RECRUITING_CANDIDATE", "maria_cruz")

from examples.demos.recruiting_distill_memory._run import run_plan  # noqa: E402


if __name__ == "__main__":
    asyncio.run(run_plan("grounded_plan_maria.json"))
