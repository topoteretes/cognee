"""Fit one rendered call inside its input token budget by dropping whole facts.

A call is trimmed in two steps. First every subject is cut to its own per-entity
fact allowance, so one crowded subject cannot crowd out the others. Then, if the
whole call still does not fit, facts are dropped worst-ranked first until it
does. Only complete facts are dropped — never their saved conflict marks and
never their source records, which stay in ``ReviewScope`` for the writer.
"""

from collections import Counter
from dataclasses import dataclass

FACT_TOKENS_PER_ENTITY = 2_000


@dataclass(frozen=True)
class TokenBudget:
    """One call's input allowance; everything but the rendered text is fixed overhead."""

    tokenizer: object
    overhead: int
    limit: int

    def fits(self, text: str) -> bool:
        return self.overhead + self.tokenizer.count_tokens(text) <= self.limit


@dataclass(frozen=True)
class _DropPlan:
    """One fixed removal order, worst ranked first; a drop count takes a prefix of it."""

    order: list[tuple[tuple, str]]
    shown_before_drops: dict[str, list[str]]

    def __len__(self) -> int:
        return len(self.order)


def fit_to_budget(call, budget: TokenBudget):
    """Shrink ``call`` until it fits, and return the rendering that does."""
    _cut_each_subject_to_its_fact_budget(call, budget.tokenizer)
    plan = _drop_plan(call)
    fits, rendered = _render_with_drops(call, plan, 0, budget)
    if fits:
        return rendered
    return _fewest_drops_that_fit(call, plan, budget)


def _count_lines_within_budget(tokenizer, lines: list[str], budget: int) -> int:
    """Per-line counts never exceed the joined count, so the running sum bounds the answer."""
    keep = 0
    total = 0
    for line in lines:
        total += tokenizer.count_tokens(line)
        if total > budget:
            break
        keep += 1
    while keep and tokenizer.count_tokens("\n".join(lines[:keep])) > budget:
        keep -= 1
    return keep


def _cut_each_subject_to_its_fact_budget(call, tokenizer) -> None:
    fact_lines = call.render().fact_lines
    for subject, fact_ids in call.shown_facts.items():
        keep = _count_lines_within_budget(tokenizer, fact_lines[subject], FACT_TOKENS_PER_ENTITY)
        del fact_ids[keep:]


def _drop_plan(call) -> _DropPlan:
    # A drop renumbers labels and can retire a context name, so only the rendered text
    # measures the call. Every subject holds its lowest ranked fact last, so the drops
    # are a prefix of one fixed order; search that order instead of re-rendering per drop.
    order = sorted(
        (
            (call.rank(subject, fact_id), subject)
            for subject, fact_ids in call.shown_facts.items()
            for fact_id in fact_ids
        ),
        reverse=True,
    )
    return _DropPlan(order, {subject: list(ids) for subject, ids in call.shown_facts.items()})


def _apply_drops(call, plan: _DropPlan, drop_count: int) -> None:
    """Keep every subject's facts minus its share of the first ``drop_count`` drops."""
    dropped = Counter(subject for _, subject in plan.order[:drop_count])
    for subject, fact_ids in plan.shown_before_drops.items():
        call.shown_facts[subject][:] = fact_ids[: len(fact_ids) - dropped[subject]]


def _render_with_drops(call, plan: _DropPlan, drop_count: int, budget: TokenBudget):
    _apply_drops(call, plan, drop_count)
    rendered = call.render()
    return budget.fits(rendered.text), rendered


def _fewest_drops_that_fit(call, plan: _DropPlan, budget: TokenBudget):
    # Doubling the probe before bisecting it keeps a call one fact over budget at two
    # renders, while one that has to shed hundreds pays a logarithmic number of them.
    low, high, best = 1, 1, None
    while best is None and low <= len(plan):
        high = min(high, len(plan))
        fits, rendered = _render_with_drops(call, plan, high, budget)
        if fits:
            best = rendered
        else:
            low, high = high + 1, high * 2
    if best is None:
        raise ValueError("Review headers exceed the input budget")
    while low < high:
        middle = (low + high) // 2
        fits, rendered = _render_with_drops(call, plan, middle, budget)
        if fits:
            high, best = middle, rendered
        else:
            low = middle + 1
    # `best` is already the rendering at `high`; only the state needs restoring.
    _apply_drops(call, plan, high)
    return best
