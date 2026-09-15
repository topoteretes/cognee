"""
Deterministic skill gate for recall().

Decides — with regexes only, no LLM and no I/O — whether a query looks
procedural enough to warrant a skill lookup. When the gate fires (and exactly
one dataset is targeted), recall() runs a metadata-only SKILLS search
concurrently with the main search and appends the hits tagged
``source="skills"``. The gate is additive: the main answer is never replaced
or blocked by it.

Disable with ``SKILL_GATE_ENABLED=false``.
"""

import os
import re
from dataclasses import dataclass, field

from cognee.api.v1.recall.query_router import _is_negated
from cognee.shared.logging_utils import get_logger

logger = get_logger("skill_gate")

# How many skills the gate lane asks for. Deliberately small: gate hits are a
# side-channel next to the main answer, not the answer itself.
DEFAULT_SKILL_GATE_TOP_K = 3

# Each rule: (pattern, weight). Weights accumulate; the gate fires at the
# threshold. Weak signals (bare ops verbs) score 2.0 so one alone never fires;
# procedural phrasings score 3.0+ and fire on their own.
_GATE_RULES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(how (do|can|should|would) (i|we|you)|how to)\b", re.IGNORECASE), 3.0),
    (re.compile(r"\b(steps? (to|for)|step.by.step)\b", re.IGNORECASE), 3.0),
    (re.compile(r"\b(procedure|playbook|runbook|checklist|workflow)\b", re.IGNORECASE), 4.0),
    (re.compile(r"\b(walk (me|us) through|guide (to|for|on))\b", re.IGNORECASE), 3.0),
    (re.compile(r"\bwhat('?s| is) the (process|procedure)\b", re.IGNORECASE), 4.0),
    (re.compile(r"\bskills?\b", re.IGNORECASE), 4.0),
    (
        re.compile(
            r"\b(set(ting)? up|setup|install(ing)?|configur(e|ing)|deploy(ing)?"
            r"|migrat(e|ing)|provision(ing)?|onboard(ing)?|rotate|troubleshoot(ing)?)\b",
            re.IGNORECASE,
        ),
        2.0,
    ),
]

_GATE_THRESHOLD = 3.0


@dataclass
class GateResult:
    """Gate decision with the score and matched fragments for observability."""

    fired: bool
    score: float = 0.0
    matched: list[str] = field(default_factory=list)


def skill_gate_enabled() -> bool:
    """``SKILL_GATE_ENABLED`` env flag; on unless explicitly disabled."""
    return os.getenv("SKILL_GATE_ENABLED", "true").strip().lower() not in ("false", "0", "no")


def should_search_skills(query: str) -> GateResult:
    """Decide whether ``query`` warrants a skill lookup. Pure function, no I/O.

    Every rule whose pattern matches (and is not negated, same suppression as
    the recall query router) adds its weight; the gate fires when the total
    reaches the threshold.
    """
    q = (query or "").strip()
    if not q:
        return GateResult(fired=False)

    score = 0.0
    matched: list[str] = []
    for pattern, weight in _GATE_RULES:
        match = pattern.search(q)
        if match and not _is_negated(q, match):
            score += weight
            matched.append(match.group(0))

    fired = score >= _GATE_THRESHOLD
    if fired:
        logger.info("skill_gate fired: score=%.1f matched=%s query=%r", score, matched, q)
    return GateResult(fired=fired, score=score, matched=matched)
