"""Deterministic rule compliance check — pure assertions, no LLM.

Reads naive_plan.json and grounded_plan.json from ./output/ and prints a
per-rule PASS/FAIL table for each, so the demo's payoff slide is literally
generated from the JSON output.
"""

import json
from pathlib import Path
from typing import Callable

HERE = Path(__file__).parent
OUTPUT = HERE / "output"


def _invite_mentions_noncompete(plan: dict) -> bool:
    invite = plan.get("screen_invite", {}) or {}
    haystack = " ".join(
        [
            invite.get("subject", ""),
            invite.get("body", ""),
            *invite.get("disclosure_questions", []),
        ]
    ).lower()
    return "non-compete" in haystack or "noncompete" in haystack or "non compete" in haystack


Predicate = Callable[[dict], bool]
PREDICATES: list[tuple[str, str, Predicate]] = [
    (
        "R1_live_coding",
        "Live coding, not take-home",
        lambda p: p["interview_format"]["format"] == "live_coding",
    ),
    (
        "R2a",
        "≥3 panelists",
        lambda p: len(p["panel"]["panelists"]) >= 3,
    ),
    (
        "R2b",
        "≥4 hours total",
        lambda p: p["panel"]["total_hours"] >= 4,
    ),
    (
        "R3_cto_on_panel",
        "Sam (CTO) on panel",
        lambda p: bool(p["panel"]["cto_included"]),
    ),
    (
        "R4_noncompete_screen",
        "Non-compete probed (Stripe)",
        _invite_mentions_noncompete,
    ),
]


def _tick(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def run() -> int:
    naive = json.loads((OUTPUT / "naive_plan.json").read_text())
    grounded = json.loads((OUTPUT / "grounded_plan.json").read_text())

    # Header
    left = "Rule"
    print(f"{left:<32} {'Naive':<10} {'Grounded':<10}")
    print("-" * 54)

    n_pass = 0
    g_pass = 0
    for _rule_id, label, pred in PREDICATES:
        n_ok = pred(naive)
        g_ok = pred(grounded)
        n_pass += int(n_ok)
        g_pass += int(g_ok)
        print(f"{label:<32} {_tick(n_ok):<10} {_tick(g_ok):<10}")

    total = len(PREDICATES)
    print("-" * 54)
    print(f"{'TOTAL':<32} {f'{n_pass}/{total}':<10} {f'{g_pass}/{total}':<10}")

    # Exit non-zero if grounded didn't pass all — makes the script also usable
    # as a CI-style check that the demo is still producing the intended payoff.
    return 0 if g_pass == total else 1


if __name__ == "__main__":
    raise SystemExit(run())
