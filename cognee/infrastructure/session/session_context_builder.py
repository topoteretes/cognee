"""Deterministic builder, ranker, and candidate applier for the active session-context layer.

This module loads stored active ``SessionContextEntry`` rows, ranks them with a deterministic
lexical ranker (no LLM), applies per-section and total character budgets, and renders a compact
four-heading guidance block. It also houses the deterministic candidate applier that turns
``CandidateContextUpdate`` items into stored entries (validate -> confidence >= 0.75 -> normalize
-> exact-dup-link-or-create, no LLM / no auto-delete).

Public orchestration coroutines are fail-open so they can never block answer generation. Pure
helpers are deliberately strict so tests catch malformed stored data and scoring mistakes.
"""

from datetime import datetime, timezone
from typing import List, Protocol, Tuple
from uuid import uuid4

from pydantic import TypeAdapter

from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    MIN_CANDIDATE_CONFIDENCE,
    AgentCandidateContextUpdateVariant,
    CandidateContextUpdate,
    ContextProfile,
    SessionContextEntry,
    normalize_content,
    valid_sections_for,
)

_AGENT_CANDIDATE_ADAPTER = TypeAdapter(AgentCandidateContextUpdateVariant)

# Default budgets. Kept conservative so the rendered block never bloats the prompt.
DEFAULT_PER_SECTION_CHAR_BUDGET = 400
DEFAULT_TOTAL_CHAR_BUDGET = 1200

# Rendered in this fixed order per profile; (section_key, heading_label).
SECTION_HEADINGS: List[Tuple[str, str]] = [
    ("goals", "Goals"),
    ("rules", "Rules"),
    ("preferences", "Preferences"),
    ("lessons_learned", "Lessons learned"),
]

AGENT_SECTION_HEADINGS: List[Tuple[str, str]] = [
    ("tool_rules", "Tool rules"),
    ("workflow_state", "Workflow state"),
    ("success_patterns", "Success patterns"),
    ("failure_lessons", "Failure lessons"),
    ("environment_facts", "Environment facts"),
]

HEADINGS_BY_PROFILE = {
    ContextProfile.QA.value: SECTION_HEADINGS,
    ContextProfile.AGENT.value: AGENT_SECTION_HEADINGS,
}


def headings_for_profile(context_profile: str) -> List[Tuple[str, str]]:
    """Ordered (section_key, heading_label) pairs for a profile; falls back to QA."""
    return HEADINGS_BY_PROFILE.get(context_profile, SECTION_HEADINGS)


BLOCK_TITLE = "## Active session guidance"
CONFLICT_INSTRUCTION = (
    "Items are listed oldest to newest within each section. "
    "When guidance conflicts, prefer the later item."
)


class ContextRanker(Protocol):
    """Thin interface for scoring an active session-context entry against a query."""

    def score(self, entry: SessionContextEntry, query: str) -> float: ...


class DeterministicRanker:
    """Pure-arithmetic ranker.

    Higher score sorts first. The score is a weighted sum of:
      * section priority (rules > goals > preferences/lessons),
      * confidence,
      * query relevance (token overlap),
      * net helpfulness (helpful_count - harmful_count),
      * recency (newer last_served_at / created_at sorts slightly higher),
      * explicit priority.

    No LLM, no I/O; deterministic for a fixed (entry, query) pair.
    """

    SECTION_PRIORITY = {"rules": 3, "goals": 2, "preferences": 1, "lessons_learned": 1}
    AGENT_SECTION_PRIORITY = {
        "failure_lessons": 3,
        "tool_rules": 3,
        "environment_facts": 2,
        "workflow_state": 2,
        "success_patterns": 1,
    }
    PRIORITY_BY_PROFILE = {
        ContextProfile.QA.value: SECTION_PRIORITY,
        ContextProfile.AGENT.value: AGENT_SECTION_PRIORITY,
    }

    # Weights chosen so section priority dominates, then confidence/relevance, then signals.
    W_SECTION = 10.0
    W_CONFIDENCE = 5.0
    W_OVERLAP = 4.0
    W_NET_HELP = 2.0
    W_PRIORITY = 1.0
    W_RECENCY = 0.5

    def _query_overlap(self, content: str, query: str) -> float:
        content_tokens = set(normalize_content(content).split())
        if not content_tokens:
            return 0.0
        query_tokens = set(normalize_content(query or "").split())
        if not query_tokens:
            return 0.0
        intersection = content_tokens & query_tokens
        return len(intersection) / len(content_tokens)

    def _recency(self, entry: SessionContextEntry) -> float:
        """Map an ISO timestamp to a small monotonic float; missing/unparseable -> 0.0."""
        stamp = entry.last_served_at or entry.created_at
        if not stamp:
            return 0.0
        try:
            parsed = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            return 0.0
        # Normalize to a small fraction so recency only tie-breaks.
        return parsed.timestamp() / 1.0e12

    def score(self, entry: SessionContextEntry, query: str) -> float:
        priority_map = self.PRIORITY_BY_PROFILE.get(entry.context_profile, self.SECTION_PRIORITY)
        section_priority = priority_map.get(entry.section, 0)
        net_help = max(-3, min(3, entry.helpful_count - entry.harmful_count))
        relevance = self._query_overlap(entry.content, query)
        return (
            self.W_SECTION * section_priority
            + self.W_CONFIDENCE * float(entry.confidence)
            + self.W_OVERLAP * relevance
            + self.W_NET_HELP * net_help
            + self.W_PRIORITY * entry.priority
            + self.W_RECENCY * self._recency(entry)
        )


def coerce_active_context_entries(raw_entries: list) -> List[SessionContextEntry]:
    """Validate stored rows and keep only active context entries."""
    entries: List[SessionContextEntry] = []
    for raw in raw_entries or []:
        if isinstance(raw, SessionContextEntry):
            entry = raw
        elif isinstance(raw, dict):
            if raw.get("kind", "context") != "context":
                continue
            entry = SessionContextEntry.model_validate(raw)
        else:
            continue
        entries.append(entry)
    return entries


def select_context_entries(
    *,
    entries: List[SessionContextEntry],
    query: str,
    ranker: ContextRanker,
    per_section_char_budget: int,
    total_char_budget: int,
    context_profile: str = ContextProfile.QA.value,
) -> List[SessionContextEntry]:
    """Select highest-scoring entries for one profile within total and per-section budgets."""
    selected: List[SessionContextEntry] = []
    section_usage = {key: 0 for key, _ in headings_for_profile(context_profile)}
    total_used = 0

    for entry in sorted(entries, key=lambda e: ranker.score(e, query), reverse=True):
        if entry.context_profile != context_profile:
            continue
        if entry.section not in section_usage:
            continue
        content = entry.content.strip()
        if not content:
            continue
        cost = len(content)
        if section_usage[entry.section] + cost > per_section_char_budget:
            continue
        if total_used + cost > total_char_budget:
            continue
        selected.append(entry)
        section_usage[entry.section] += cost
        total_used += cost

    return selected


def _time_label(timestamp: str) -> str:
    """Return a compact HH:MM:SS label from an ISO timestamp; invalid/missing -> unknown."""
    if not timestamp:
        return "unknown"
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return "unknown"


def _render_entry(entry: SessionContextEntry) -> str:
    return f"[{_time_label(entry.created_at)}] {entry.content.strip()}"


def _render_block(grouped_rendered: List[Tuple[str, List[str]]]) -> str:
    """Assemble the final block string from (heading_label, [bullet_lines]) groups."""
    lines: List[str] = [BLOCK_TITLE, CONFLICT_INSTRUCTION]
    for heading_label, bullets in grouped_rendered:
        if not bullets:
            continue
        lines.append(f"### {heading_label}")
        lines.extend(f"- {bullet}" for bullet in bullets)
    return "\n".join(lines)


async def _stamp_served_entries(*, session_manager, user_id, session_id, entry_ids: List[str]):
    """Best-effort stamp for entries actually rendered into the active context block."""
    if not entry_ids:
        return

    served_at = datetime.now(timezone.utc).isoformat()
    for entry_id in entry_ids:
        try:
            await session_manager.update_session_context_entry(
                user_id=user_id,
                session_id=session_id,
                entry_id=entry_id,
                merge={"last_served_at": served_at},
            )
        except Exception:
            continue


async def build_active_context_block(
    *,
    session_manager,
    user_id,
    session_id,
    query,
    ranker: ContextRanker | None = None,
    per_section_char_budget: int = DEFAULT_PER_SECTION_CHAR_BUDGET,
    total_char_budget: int = DEFAULT_TOTAL_CHAR_BUDGET,
    context_profile: str = ContextProfile.QA.value,
    stamp_served: bool = True,
) -> Tuple[str, List[str]]:
    """Load active context entries, rank them, budget-cap, and render a compact block.

    Renders only ``context_profile`` entries. With ``stamp_served=False`` the call performs
    no writes (used by read-only recall); the mutating default stamps ``last_served_at`` on
    the rendered entries. Returns ``(block_str, served_ids)`` — the entry ids rendered into the
    block — or ``("", [])`` on any exception (fail-open).
    """
    try:
        if ranker is None:
            ranker = DeterministicRanker()

        raw_entries = await session_manager.get_session_context_entries(
            user_id=user_id,
            session_id=session_id,
        )
        if not raw_entries:
            return "", []

        entries = coerce_active_context_entries(raw_entries)
        if not entries:
            return "", []

        selected = select_context_entries(
            entries=entries,
            query=query,
            ranker=ranker,
            per_section_char_budget=per_section_char_budget,
            total_char_budget=total_char_budget,
            context_profile=context_profile,
        )
        if not selected:
            return "", []

        headings = headings_for_profile(context_profile)
        by_section = {key: [] for key, _ in headings}
        for entry in selected:
            by_section[entry.section].append(entry)

        grouped_rendered: List[Tuple[str, List[str]]] = []
        for section_key, heading_label in headings:
            section_entries = sorted(
                by_section.get(section_key, []),
                key=lambda entry: entry.created_at or "",
            )
            bullets = [_render_entry(entry) for entry in section_entries]
            grouped_rendered.append((heading_label, bullets))

        served_ids = [entry.id for entry in selected]
        block = _render_block(grouped_rendered)
        if stamp_served:
            await _stamp_served_entries(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                entry_ids=served_ids,
            )
        return block, served_ids
    except Exception:
        # Fail-open: never block answer generation.
        return "", []


async def _apply_single_candidate(
    *,
    session_manager,
    user_id,
    session_id,
    source_id,
    candidate: CandidateContextUpdate,
) -> str | None:
    """Apply one validated candidate. Returns the touched/created entry id, or None to skip.

    Implements validate -> confidence>=0.75 -> normalize -> dup-link-or-create, where a dup is
    an exact normalized-content match within the same (profile, section). ``source_id`` is the
    evidence id linked on the entry — a feedback id for qa, a trace id for agent.
    Raises on store errors; the caller wraps this so one failure never aborts the batch.
    """
    context_profile = getattr(candidate, "context_profile", ContextProfile.QA.value)
    section = candidate.section
    if section not in valid_sections_for(context_profile):
        return None

    content = candidate.content.strip()
    if not content:
        return None
    if len(content) > MAX_CONTEXT_CONTENT_CHARS:
        return None
    if float(candidate.confidence) < MIN_CANDIDATE_CONFIDENCE:
        return None

    normalized = normalize_content(content)
    source_field = (
        "source_trace_ids"
        if context_profile == ContextProfile.AGENT.value
        else "source_feedback_ids"
    )

    # Look for an existing entry in the same (profile, section) with identical normalized content.
    existing = await session_manager.get_session_context_entries(
        user_id=user_id,
        session_id=session_id,
    )
    duplicate_row = None
    for raw in existing or []:
        if isinstance(raw, SessionContextEntry):
            row = raw.model_dump()
        elif isinstance(raw, dict):
            row = raw
        else:
            continue
        if row.get("kind", "context") != "context":
            continue
        if row.get("context_profile", ContextProfile.QA.value) != context_profile:
            continue
        if row.get("section") != section:
            continue
        if row.get("normalized_content") == normalized:
            duplicate_row = row
            break

    if duplicate_row is not None:
        entry_id = duplicate_row.get("id")
        source_ids = list(duplicate_row.get(source_field) or [])
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
            await session_manager.update_session_context_entry(
                user_id=user_id,
                session_id=session_id,
                entry_id=entry_id,
                merge={source_field: source_ids},
            )
        return entry_id

    # Novel content: create a new active entry under this profile/section.
    linked_ids = [source_id] if source_id else []
    new_entry = SessionContextEntry(
        id=str(uuid4()),
        section=section,
        context_profile=context_profile,
        content=content,
        normalized_content=normalized,
        confidence=float(candidate.confidence),
        created_at=datetime.now(timezone.utc).isoformat(),
        source_feedback_ids=linked_ids if source_field == "source_feedback_ids" else [],
        source_trace_ids=linked_ids if source_field == "source_trace_ids" else [],
        kind="context",
    )
    await session_manager.create_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_dump=new_entry.model_dump(),
    )
    return new_entry.id


def _coerce_candidate_model(candidate) -> CandidateContextUpdate:
    """Validate a raw candidate (dict or model) into the right profile's candidate model."""
    if isinstance(candidate, CandidateContextUpdate):
        return candidate
    if (
        isinstance(candidate, dict)
        and candidate.get("context_profile") == ContextProfile.AGENT.value
    ):
        return _AGENT_CANDIDATE_ADAPTER.validate_python(candidate)
    return CandidateContextUpdate.model_validate(candidate)


async def apply_candidate_updates(
    *,
    session_manager,
    user_id,
    session_id,
    source_id,
    candidates: list,
) -> List[str]:
    """Deterministically apply candidate context updates (qa or agent).

    For each candidate: validate section/content/length for its profile, require
    confidence >= 0.75, normalize, then either link to an exact duplicate (append ``source_id``
    to its source list) or create a new entry. ``source_id`` is a feedback id for qa candidates
    and a trace id for agent candidates. No LLM, no fuzzy matching, no auto-delete. Each
    candidate is wrapped so one failure never raises; the whole call is fail-open and returns
    ``[]`` on outer failure.

    Returns the list of touched/created entry ids.
    """
    touched: List[str] = []
    try:
        for candidate in candidates or []:
            try:
                model = _coerce_candidate_model(candidate)
                entry_id = await _apply_single_candidate(
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                    source_id=source_id,
                    candidate=model,
                )
                if entry_id:
                    touched.append(entry_id)
            except Exception:
                # Per-candidate fail-open: skip this candidate, keep going.
                continue
        return touched
    except Exception:
        return touched
