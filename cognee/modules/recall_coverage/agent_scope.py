"""Resolve an ``agent_label`` into a session-id scope.

An agent is ``(API key owner, session-id prefix)``: a tool such as Claude Code
or Codex stamps a known prefix on every session it opens, so "the questions
Claude Code asked" is exactly "the recalls whose ``session_id`` starts with one
of Claude Code's prefixes".

Three ``LIKE`` traps live here rather than in the query builder, because a raw
prefix must never reach a pattern:

1. ``_`` is a single-character wildcard and ``%`` matches anything, so
   ``LIKE 'claude_%'`` also matches every ``claude_desktop_...`` session — Claude
   Desktop's traffic would be reported as Claude Code's. Prefixes are escaped
   here with :func:`escape_like_prefix`; the matching side pairs them with
   ``ESCAPE '\\'`` (see
   :mod:`cognee.modules.search.operations.get_queries`).
2. One label maps to one *or more* prefixes, so a label's predicate is an OR
   group and never a single comparison.
3. Longest prefix wins, so ``claude_desktop_`` must be tested before
   ``claude_``. Escaping does not help here — ``claude_desktop_a1`` really does
   start with the literal ``claude_`` — so a label's scope also carries the
   longer prefixes other labels own, for its predicate to subtract. Prefixes are
   sorted longest-first by the raw prefix, which is the order a classifier walks.

Two directions, one rule. :func:`resolve_agent_scope` turns a label into a SQL
predicate ("which rows are this agent's"); :func:`classify_session` turns one
session id into a label ("which agent is this row's"), which is what the per-row
``agent`` column needs. Both walk :func:`_longest_first`, so the two can only
ever agree — a second copy of "longest prefix wins" would drift, and the visible
symptom would be a row filed under a label whose own predicate does not select it.

Nothing downstream takes a raw label string: :func:`resolve_agent_scope`
validates it once, rejects a typo, and everything else takes the resulting
:class:`AgentScope`.
"""

from typing import Any, Iterable, Optional

from cognee.modules.recall_coverage.config import (
    RecallCoverageConfig,
    get_recall_coverage_config,
)
from cognee.modules.recall_coverage.exceptions import UnknownAgentLabelError
from cognee.modules.recall_coverage.types import (
    LIKE_ESCAPE_CHAR,
    AgentScope,
    AgentScopeMode,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# The three reserved labels.
#
# ``api`` — an unknown prefix, or no session at all: the complement of the whole
# prefix map. Its scope carries every known prefix so the query can negate them.
# ``all`` — no session predicate whatsoever. A first-class permanent mode ("how
# is my memory doing overall"), also the default when the request omits a label.
# ``ui`` — a human in the cloud search box. Reserved as a *name*, but unlike the
# other two it resolves through the ordinary prefix map (see
# ``config.DEFAULT_AGENT_PREFIX_MAP``, which gives it ``search-ui-`` — what the
# shipped frontend mints — and the reserved ``ui_``). It deliberately gets no
# scope of its own: a complement-of-the-map scope is exactly ``api``, and two
# labels sharing one predicate would count the same traffic twice.
#
# Honest caveat about ``ui`` today: the shipped frontend logs its recalls under
# ``query_type = "HYBRID_COMPLETION"``, which is not one of the six recall types
# in ``config.DEFAULT_QUERY_TYPES``, so a ``ui`` run resolves and validates but
# selects no rows until that changes. The label is still correct to accept —
# rejecting it would make a real, prefix-distinguishable surface unnameable.
LABEL_API = "api"
LABEL_ALL = "all"
LABEL_UI = "ui"

# The characters that mean something other than themselves inside a LIKE
# pattern. The escape character comes first: escaping it after the wildcards
# would double the backslashes this function just introduced.
_LIKE_METACHARACTERS = (LIKE_ESCAPE_CHAR, "_", "%")


def escape_like_prefix(prefix: str) -> str:
    """Escape a session-id prefix for use in a ``LIKE`` pattern.

    Every prefix in the map ends in ``_``, so this is not a corner case: without
    it, ``cc_`` matches ``ccx...`` and ``claude_`` matches ``claude_desktop_...``.
    Pair the result with ``ESCAPE '\\'`` — an escape character is only honoured
    when the statement declares it.
    """
    escaped = prefix
    for metacharacter in _LIKE_METACHARACTERS:
        escaped = escaped.replace(metacharacter, LIKE_ESCAPE_CHAR + metacharacter)
    return escaped


def _longest_first(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate and order longest **raw** prefix first.

    The one place "longest prefix wins" is expressed. Both directions read it:
    the SQL side escapes the result (:func:`_escaped_longest_first`), the Python
    classifier walks it as-is (:func:`classify_session`). The secondary sort key
    only makes the order deterministic for equal-length prefixes.
    """
    return tuple(sorted(set(prefixes), key=lambda prefix: (-len(prefix), prefix)))


def _escaped_longest_first(prefixes: Iterable[str]) -> tuple[str, ...]:
    """:func:`_longest_first`, escaped for ``LIKE``.

    Ordered before escaping because "longest wins" is a statement about the raw
    prefixes; escaping inflates lengths unevenly (``a_`` becomes as long as
    ``abc``) and would reshuffle them.
    """
    return tuple(escape_like_prefix(prefix) for prefix in _longest_first(prefixes))


def resolve_agent_scope(
    agent_label: Optional[str] = None,
    user: Optional[Any] = None,
    config: Optional[RecallCoverageConfig] = None,
) -> AgentScope:
    """Validate ``agent_label`` and return the scope every query builder takes.

    An unknown label raises :class:`UnknownAgentLabelError` (422) instead of
    resolving to no prefixes: an empty report is a legitimate answer for a real
    label, so a typo silently producing one would be indistinguishable from
    "this agent has asked nothing yet". ``user`` is only used to name the caller
    in that warning, which is what makes a bad label traceable to a client.
    """
    if config is None:
        config = get_recall_coverage_config()
    prefix_map = config.prefix_map()

    label = (agent_label or "").strip() or LABEL_ALL

    if label == LABEL_ALL:
        return AgentScope(label=LABEL_ALL, prefixes=(), mode=AgentScopeMode.ALL)

    if label == LABEL_API:
        return AgentScope(
            label=LABEL_API,
            prefixes=_escaped_longest_first(
                prefix for prefixes in prefix_map.values() for prefix in prefixes
            ),
            mode=AgentScopeMode.NEGATED,
        )

    if label not in prefix_map:
        logger.warning(
            "recall_coverage: unknown agent label %r requested by user %s",
            label,
            getattr(user, "id", None),
        )
        raise UnknownAgentLabelError(message=f"Unknown agent label: {label}")

    return AgentScope(
        label=label,
        prefixes=_escaped_longest_first(prefix_map[label]),
        mode=AgentScopeMode.PREFIX,
        excluded_prefixes=_overridden_by(label, prefix_map),
    )


def _overridden_by(label: str, prefix_map: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Prefixes of *other* labels that extend one of ``label``'s own.

    ``claude-code`` owns ``claude_`` and ``claude-desktop`` owns
    ``claude_desktop_``, so every Claude Desktop session is also a literal
    ``claude_`` match. Longest prefix wins, so those sessions belong to Claude
    Desktop and Claude Code's predicate must subtract them.

    A label's own prefixes are never subtracted, so a map that gives one label
    both ``a_`` and ``a_b_`` still matches all of its own sessions.
    """
    own = prefix_map[label]
    overriding = {
        prefix
        for other_label, prefixes in prefix_map.items()
        if other_label != label
        for prefix in prefixes
        if prefix not in own and any(prefix.startswith(base) and prefix != base for base in own)
    }
    return _escaped_longest_first(overriding)


def classify_session(
    session_id: Optional[str], config: Optional[RecallCoverageConfig] = None
) -> str:
    """The label one recall's ``session_id`` belongs to. Never raises.

    This is the per-row half of attribution: a run's own label is usually ``all``,
    so "which agent asked this" has to be decided per row rather than inherited
    from the run.

    Walks the raw prefixes longest-first, so ``claude_desktop_a1`` is Claude
    Desktop and ``claude_a1`` is Claude Code. Raw, not escaped: the escaped
    prefixes on :class:`AgentScope` are ``LIKE`` patterns, and since every default
    prefix ends in ``_``, ``startswith`` against one of those would match nothing.

    Falls back to ``api`` — including for ``None``, an empty string, and an
    unrecognised prefix. That is not a default, it is the definition: ``api`` is
    the complement of the map, and its SQL predicate selects exactly "no session
    at all, or a session no known prefix claims". The invariant worth testing is
    that agreement: ``classify_session(s) == L`` iff the predicate
    ``build_session_predicate`` builds for ``L`` matches ``s``. Never returns
    ``all``, which is not a classification but the absence of one.
    """
    if config is None:
        config = get_recall_coverage_config()

    prefix_map = config.prefix_map()
    label_of = {prefix: label for label, prefixes in prefix_map.items() for prefix in prefixes}

    text = session_id or ""
    if not text:
        return LABEL_API

    for prefix in _longest_first(label_of):
        if text.startswith(prefix):
            return label_of[prefix]

    return LABEL_API
