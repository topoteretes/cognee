"""Deterministic rule compliance check — pure assertions, no LLM.

Reads every {naive,grounded}_plan_<candidate>.json file in output/ and
prints a per-candidate PASS/FAIL matrix across the six Ledgerline rules.

Predicates that don't apply to a given candidate (e.g. R4 only fires
for ex-Stripe/Plaid/Adyen) render as '--' and don't count in the
per-candidate total.
"""

import json
import re
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).parent
OUTPUT = HERE / "output"

EXPECTED_PANEL_FIRST_NAMES = {"Sam", "Jordan", "Leila", "Ravi"}
STRIPE_LIKE = {"Stripe", "Plaid", "Adyen"}


def _panelists_match_exact(plan: dict) -> bool:
    panelists = (plan.get("panel") or {}).get("panelists") or []
    if len(panelists) != 4:
        return False
    # First whitespace/em-dash/hyphen-delimited token is the person's first name.
    names = {re.split(r"\s|—|-", p.strip(), maxsplit=1)[0] for p in panelists}
    return names == EXPECTED_PANEL_FIRST_NAMES


def _body_mentions_streamtap(plan: dict) -> bool:
    body = (plan.get("screen_invite") or {}).get("body") or ""
    return "streamtap" in body.lower()


def _mentions_noncompete(text: str) -> bool:
    # LLMs often use typographic dashes (U+2010..U+2015, U+2212) instead of
    # ASCII hyphen-minus. Normalize so the check is robust.
    t = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "―", "−"):
        t = t.replace(dash, "-")
    return "non-compete" in t or "noncompete" in t or "non compete" in t


def _noncompete_is_first(plan: dict) -> bool:
    qs = (plan.get("screen_invite") or {}).get("disclosure_questions") or []
    return bool(qs) and _mentions_noncompete(qs[0])


def _applies_r4(plan: dict) -> bool:
    return (plan.get("candidate") or {}).get("prior_company") in STRIPE_LIKE


Predicate = Callable[[dict], Optional[bool]]

# (rule_id, short label, predicate). Predicate returns None when the rule
# does not apply to this candidate (so it's not counted for/against).
PREDICATES: list[tuple[str, str, Predicate]] = [
    (
        "R1a",
        "Format == live_coding",
        lambda p: (p.get("interview_format") or {}).get("format") == "live_coding",
    ),
    (
        "R1b",
        "Duration == 80 min",
        lambda p: (p.get("interview_format") or {}).get("duration_minutes") == 80,
    ),
    (
        "R2",
        "Panel == {Sam,Jordan,Leila,Ravi}",
        _panelists_match_exact,
    ),
    (
        "R3",
        "Medium == onsite",
        lambda p: (p.get("interview_format") or {}).get("medium") == "onsite",
    ),
    (
        "R4",
        "Non-compete is FIRST question",
        lambda p: _noncompete_is_first(p) if _applies_r4(p) else None,
    ),
    (
        "R5",
        "Body mentions 'streamtap'",
        _body_mentions_streamtap,
    ),
    (
        "R6",
        "total_hours == 4.0",
        lambda p: (p.get("panel") or {}).get("total_hours") == 4.0,
    ),
]


def _tick(val: Optional[bool]) -> str:
    if val is None:
        return "--"
    return "PASS" if val else "FAIL"


def _load_plans() -> dict[str, dict[str, dict]]:
    """Return {candidate: {mode: plan_dict}} for every plan JSON in output/."""
    plans: dict[str, dict[str, dict]] = {}
    for path in sorted(OUTPUT.glob("*_plan_*.json")):
        stem = path.stem
        for mode in ("naive", "grounded"):
            prefix = f"{mode}_plan_"
            if stem.startswith(prefix):
                candidate = stem[len(prefix):]
                plans.setdefault(candidate, {})[mode] = json.loads(path.read_text())
                break
    return plans


def run() -> int:
    plans = _load_plans()
    if not plans:
        print("No plan files found under output/. Run run_matrix.py first.")
        return 1

    candidates = sorted(plans.keys())
    # Column widths: 6-char "PASS" / "FAIL" / "--" with padding.
    label_w = 34
    cell_w = 8

    # Header row.
    header = f"{'Rule':<{label_w}}"
    for cand in candidates:
        header += f" | {cand + ' N':<{cell_w}}{cand + ' G':<{cell_w}}"
    print(header)
    print("-" * len(header))

    totals_naive = {c: [0, 0] for c in candidates}      # [passes, applicable]
    totals_grounded = {c: [0, 0] for c in candidates}

    for _rid, label, pred in PREDICATES:
        row = f"{label:<{label_w}}"
        for cand in candidates:
            naive = plans[cand].get("naive")
            grounded = plans[cand].get("grounded")
            n_ok = pred(naive) if naive else None
            g_ok = pred(grounded) if grounded else None
            if n_ok is not None:
                totals_naive[cand][1] += 1
                totals_naive[cand][0] += int(n_ok)
            if g_ok is not None:
                totals_grounded[cand][1] += 1
                totals_grounded[cand][0] += int(g_ok)
            row += f" | {_tick(n_ok):<{cell_w}}{_tick(g_ok):<{cell_w}}"
        print(row)

    print("-" * len(header))
    totals_row = f"{'TOTAL (per candidate)':<{label_w}}"
    for cand in candidates:
        n_pass, n_total = totals_naive[cand]
        g_pass, g_total = totals_grounded[cand]
        totals_row += (
            f" | {f'{n_pass}/{n_total}':<{cell_w}}{f'{g_pass}/{g_total}':<{cell_w}}"
        )
    print(totals_row)

    # Aggregate across all candidates.
    all_n_pass = sum(t[0] for t in totals_naive.values())
    all_n_total = sum(t[1] for t in totals_naive.values())
    all_g_pass = sum(t[0] for t in totals_grounded.values())
    all_g_total = sum(t[1] for t in totals_grounded.values())
    print()
    print(
        f"Aggregate across {len(candidates)} candidates: "
        f"naive {all_n_pass}/{all_n_total}, grounded {all_g_pass}/{all_g_total}"
    )

    return 0 if all_g_pass == all_g_total else 1


if __name__ == "__main__":
    raise SystemExit(run())
