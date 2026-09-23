"""Pure weight math for user preferences — no I/O, no config reads.

This is the one home for the preference weight arithmetic: the decay
(``effective_weight``) and the retrieval-time ranking factor
(``personal_factor``). Each is called from the update in ``update.py`` and
the retrieval lookup only, so exactly one place knows each formula.
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


def personal_factor(weight: float, influence: float, *, distance_space: bool) -> float:
    """Multiplicative ranking factor for one personal ``prefers`` weight.

    Exactly 1.0 at a neutral weight (0.5) and at zero influence, so a no-op is
    arithmetically exact. ``influence`` reads as "the most personalization may
    move a score": 0.3 means at most 30%. In distance space (lower is better)
    a preferred item's distance shrinks; in score space (higher is better) its
    score grows — the same signal, mirrored per ranking convention.
    """
    signal = float(influence) * (2.0 * float(weight) - 1.0)
    return (1.0 - signal) if distance_space else (1.0 + signal)
