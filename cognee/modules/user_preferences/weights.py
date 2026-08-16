"""Pure weight math for user preferences — no I/O, no config reads.

This is the one home for the preference weight arithmetic: the decay below,
and (from Phase 3) the retrieval-time ``personal_factor``. It is called from
exactly two places — the update in ``update.py`` and the retrieval lookup —
so exactly one place knows each formula.
"""


def effective_weight(
    weight: float,
    updated_at_turn: int,
    turn_counter: int,
    beta: float,
) -> float:
    """A stored weight pulled back toward neutral for every turn it sat idle.

    The convex evidence update ``w + alpha * (target - w)`` rewrites as
    ``(1 - alpha) * w + alpha * target``; this is the same operation with
    neutral (0.5) as the target, applied once per idle turn. Nothing ever
    rewrites a row just to age it — decay happens on read, from the turn
    counter, so there is no clock and no timestamp anywhere in the math.
    """
    idle = max(0, int(turn_counter) - int(updated_at_turn))
    return 0.5 + (float(weight) - 0.5) * (1.0 - float(beta)) ** idle
