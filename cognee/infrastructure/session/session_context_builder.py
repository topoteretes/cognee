"""Deterministic builder, ranker, and candidate applier for the active session-context layer.

This module loads stored active ``SessionContextEntry`` rows, ranks them with a pure-arithmetic
deterministic ranker (no LLM), applies per-section and total character budgets, and renders a
compact four-heading guidance block. It also houses the deterministic candidate applier that
turns ``CandidateContextUpdate`` items into stored entries (validate -> confidence >= 0.75 ->
normalize -> exact-dup-link-or-create, no LLM / no fuzzy / no auto-delete).

Every public coroutine is fail-open: any exception degrades to a no-op safe default
(``("", [])`` for the builder, ``[]`` for the applier) so it can never block answer generation.
"""

from datetime import datetime
from typing import List, Protocol, Tuple
from uuid import uuid4

from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    MIN_CANDIDATE_CONFIDENCE,
    VALID_SECTIONS,
    CandidateContextUpdate,
    SessionContextEntry,
    normalize_content,
)

# Default budgets. Kept conservative so the rendered block never bloats the prompt.
DEFAULT_PER_SECTION_CHAR_BUDGET = 400
DEFAULT_TOTAL_CHAR_BUDGET = 1200

# Rendered in this fixed order; (section_key, heading_label).
SECTION_HEADINGS: List[Tuple[str, str]] = [
    ("goals", "Goals"),
    ("rules", "Rules"),
    ("preferences", "Preferences"),
    ("lessons_learned", "Lessons learned"),
]

BLOCK_TITLE = "## Active session guidance"


class ContextRanker(Protocol):
    """Thin interface for scoring an active session-context entry against a query."""

    def score(self, entry: SessionContextEntry, query: str) -> float: ...


class DeterministicRanker:
    """Pure-arithmetic ranker.

    Higher score sorts first. The score is a weighted sum of:
      * section priority (rules > goals > preferences/lessons),
      * confidence,
      * query-term overlap (fraction of entry tokens that appear in the query),
      * net helpfulness (helpful_count - harmful_count),
      * recency (newer last_served_at / created_at sorts slightly higher),
      * explicit priority.

    No LLM, no I/O; deterministic for a fixed (entry, query) pair.
    """

    SECTION_PRIORITY = {"rules": 3, "goals": 2, "preferences": 1, "lessons_learned": 1}

    # Weights chosen so section priority dominates, then confidence/overlap, then signals.
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
        section_priority = self.SECTION_PRIORITY.get(entry.section, 0)
        net_help = entry.helpful_count - entry.harmful_count
        overlap = self._query_overlap(entry.content, query)
        return (
            self.W_SECTION * section_priority
            + self.W_CONFIDENCE * float(entry.confidence)
            + self.W_OVERLAP * overlap
            + self.W_NET_HELP * net_help
            + self.W_PRIORITY * entry.priority
            + self.W_RECENCY * self._recency(entry)
        )


def _render_block(grouped_rendered: List[Tuple[str, List[str]]]) -> str:
    """Assemble the final block string from (heading_label, [bullet_lines]) groups."""
    lines: List[str] = [BLOCK_TITLE]
    for heading_label, bullets in grouped_rendered:
        if not bullets:
            continue
        lines.append(f"### {heading_label}")
        lines.extend(f"- {bullet}" for bullet in bullets)
    return "\n".join(lines)


async def build_active_context_block(
    *,
    session_manager,
    user_id,
    session_id,
    query,
    ranker: ContextRanker = None,
    per_section_char_budget: int = DEFAULT_PER_SECTION_CHAR_BUDGET,
    total_char_budget: int = DEFAULT_TOTAL_CHAR_BUDGET,
) -> Tuple[str, List[str]]:
    """Load active context entries, rank them, budget-cap, and render a compact block.

    Returns ``(block_str, served_ids)`` where ``served_ids`` lists the entry ids actually
    rendered into the block. Returns ``("", [])`` on any exception (fail-open).
    """
    try:
        if ranker is None:
            ranker = DeterministicRanker()

        raw_entries = await session_manager.get_session_context_entries(user_id, session_id)
        if not raw_entries:
            return "", []

        # Coerce dict payloads into validated models; skip anything that fails / is feedback.
        entries: List[SessionContextEntry] = []
        for raw in raw_entries:
            try:
                if isinstance(raw, SessionContextEntry):
                    entry = raw
                elif isinstance(raw, dict):
                    if raw.get("kind", "context") != "context":
                        continue
                    entry = SessionContextEntry.model_validate(raw)
                else:
                    continue
            except Exception:
                continue
            if entry.kind != "context":
                continue
            entries.append(entry)

        if not entries:
            return "", []

        # Group by section and rank within section (highest score first).
        by_section = {key: [] for key, _ in SECTION_HEADINGS}
        for entry in entries:
            if entry.section in by_section:
                by_section[entry.section].append(entry)
        for section_entries in by_section.values():
            section_entries.sort(key=lambda e: ranker.score(e, query), reverse=True)

        served_ids: List[str] = []
        grouped_rendered: List[Tuple[str, List[str]]] = []
        total_used = 0

        for section_key, heading_label in SECTION_HEADINGS:
            section_used = 0
            bullets: List[str] = []
            for entry in by_section.get(section_key, []):
                bullet = entry.content.strip()
                if not bullet:
                    continue
                cost = len(bullet)
                if section_used + cost > per_section_char_budget:
                    continue
                if total_used + cost > total_char_budget:
                    continue
                bullets.append(bullet)
                served_ids.append(entry.id)
                section_used += cost
                total_used += cost
            grouped_rendered.append((heading_label, bullets))

        if not served_ids:
            return "", []

        return _render_block(grouped_rendered), served_ids
    except Exception:
        # Fail-open: never block answer generation.
        return "", []


async def _apply_single_candidate(
    *,
    session_manager,
    user_id,
    session_id,
    feedback_entry_id,
    candidate: CandidateContextUpdate,
) -> str | None:
    """Apply one validated candidate. Returns the touched/created entry id, or None to skip.

    Implements validate -> confidence>=0.75 -> normalize -> exact-dup-link-or-create.
    Raises on store errors; the caller wraps this so one failure never aborts the batch.
    """
    section = candidate.section
    if section not in VALID_SECTIONS:
        return None

    content = candidate.content.strip()
    if not content:
        return None
    if len(content) > MAX_CONTEXT_CONTENT_CHARS:
        return None
    if float(candidate.confidence) < MIN_CANDIDATE_CONFIDENCE:
        return None

    normalized = normalize_content(content)

    # Look for an existing active entry with the same section + normalized content (exact dup).
    existing = await session_manager.get_session_context_entries(user_id, session_id)
    for raw in existing or []:
        if isinstance(raw, SessionContextEntry):
            row = raw.model_dump()
        elif isinstance(raw, dict):
            row = raw
        else:
            continue
        if row.get("kind", "context") != "context":
            continue
        if row.get("section") == section and row.get("normalized_content") == normalized:
            entry_id = row.get("id")
            source_ids = list(row.get("source_feedback_ids") or [])
            if feedback_entry_id and feedback_entry_id not in source_ids:
                source_ids.append(feedback_entry_id)
                await session_manager.update_session_context_entry(
                    user_id,
                    session_id,
                    entry_id,
                    {"source_feedback_ids": source_ids},
                )
            return entry_id

    # Novel content: create a new active entry.
    new_entry = SessionContextEntry(
        id=str(uuid4()),
        section=section,
        content=content,
        normalized_content=normalized,
        confidence=float(candidate.confidence),
        created_at=datetime.utcnow().isoformat(),
        source_feedback_ids=[feedback_entry_id] if feedback_entry_id else [],
        kind="context",
    )
    await session_manager.create_session_context_entry(user_id, session_id, new_entry.model_dump())
    return new_entry.id


async def apply_candidate_updates(
    *,
    session_manager,
    user_id,
    session_id,
    feedback_entry_id,
    candidates: list,
    served_rating_ids: list = None,
) -> List[str]:
    """Deterministically apply candidate context updates.

    For each candidate: validate section/content/length, require confidence >= 0.75, normalize,
    then either link to an exact duplicate (append the feedback id to its source list) or create
    a new entry. No LLM, no fuzzy matching, no auto-delete. Each candidate is wrapped so one
    failure never raises; the whole call is fail-open and returns ``[]`` on outer failure.

    Returns the list of touched/created entry ids.
    """
    touched: List[str] = []
    try:
        for candidate in candidates or []:
            try:
                model = (
                    candidate
                    if isinstance(candidate, CandidateContextUpdate)
                    else CandidateContextUpdate.model_validate(candidate)
                )
                entry_id = await _apply_single_candidate(
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                    feedback_entry_id=feedback_entry_id,
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
