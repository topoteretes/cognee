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
    # Chinese (CJK) rules. These deliberately avoid \b: Chinese is written
    # without spaces, so a sentence is one continuous \w run and \b can never
    # match inside it — English-only rules score every Chinese query 0.0.
    # Tiers mirror the English ones: procedural phrasings that fire on their
    # own (3.0), strong document nouns (4.0), and weak signals (2.0) that only
    # fire in combination.
    # Interrogative + ops verb within a short window ("怎么配置…", "如何部署…",
    # "怎么排查这个报错"), like "how do I install…".
    (
        re.compile(
            r"(?:怎么|怎样|如何|咋)[^？?！!。]{0,12}?"
            r"(?:配置|设置|安装|部署|迁移|搭建|排查|调试|升级|接入|集成|运行|启用|卸载|修复)"
        ),
        3.0,
    ),
    # Request + ops verb ("帮我配置…", "麻烦帮忙排查…"), also firing alone.
    (
        re.compile(
            r"(?:帮我|帮忙|请帮我|麻烦)[^？?！!。]{0,12}?"
            r"(?:配置|设置|安装|部署|迁移|搭建|排查|调试|升级|接入|集成|运行|修复|检查|卸载)"
        ),
        3.0,
    ),
    # Strong document nouns, like procedure|playbook|runbook|checklist.
    (
        re.compile(
            r"(?:操作手册|运维手册|运行手册|操作指南|配置指南|安装指南|部署指南"
            r"|排查指南|使用指南|检查清单|操作流程|操作规范|运维流程)"
        ),
        4.0,
    ),
    # Bare ops verbs and 步骤 (steps): weak signals, one alone never fires —
    # mirroring the English verb rule. They stack with each other or with the
    # phrasing rules above.
    (
        re.compile(r"(?:安装|部署|配置|迁移|排查|调试|搭建|升级|卸载|初始化|集成|接入|修复|步骤)"),
        2.0,
    ),
]

_GATE_THRESHOLD = 3.0

# Suppress a match if a negation word appears within this many characters
# before the match start.
# ``n't`` is a suffix, not a word: there is no boundary between the "o" and
# the "n" of "don't", so it needs its own alternative outside the group.
# The Chinese alternatives are also boundary-free (CJK text has no word
# boundaries at all). 没 is deliberately absent: it also occurs inside the
# affirmative-interrogative 有没有 ("is there…"), which would suppress half
# of all Chinese questions.
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|lack)\b|n't\b|(?:不|别|勿|未|无需|无须)",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 20


def _is_negated(query: str, match: re.Match) -> bool:
    """True if a negation word sits just before this regex match."""
    start = max(0, match.start() - _NEGATION_WINDOW)
    prefix = query[start : match.start()]
    return bool(_NEGATION.search(prefix))


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

    Every rule whose pattern matches (and is not negated) adds its weight;
    the gate fires when the total reaches the threshold.
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
