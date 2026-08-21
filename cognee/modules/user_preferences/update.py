"""Fold a user's rated turns and stated preferences into their preference subgraph.

Modeled on ``session_distillation/distill.py``: a plain async function that
reads session rows, computes, and writes a small targeted subgraph — not a
pipeline. Runs as a stage of ``improve()``, once for all sessions in scope,
because preferences aggregate across a user's sessions.

Weights and text are two independent tracks: a user can rate turns without
stating a preference, and can state a preference without rating anything.
The rating steps run only over turns that carried a usable rating; the clock,
the prune pass, and the text refresh run on every call. There is no early
return anywhere.

Hard rules honoured here:

- ``SessionQAEntry.feedback_score`` is only ever *read*, never written — it
  gates ``apply_feedback_weights``, which moves global weights.
- Only ``node_ids`` become ``prefers`` targets; ``edge_ids`` are ignored,
  because an edge cannot be the endpoint of an edge.
- Decay is never written: it is computed on read via ``effective_weight``
  from a turn counter that advances on every turn, rated or not.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from pydantic import BaseModel

from cognee.base_config import get_base_config
from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_context_builder import coerce_active_context_entries
from cognee.infrastructure.session.session_context_models import (
    ContextSection,
    SessionContextEntry,
    is_context_entry_usable,
)
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.apply_feedback_weights import (
    normalize_feedback_score,
    stream_update_weight,
)

from .constants import (
    MAX_PREFERENCE_TEXT_CHARS,
    NEUTRAL_WEIGHT,
    PREFERENCE_DELETE_THRESHOLD,
    PREFERENCE_TURN_COUNTED_KEY,
    PREFERENCE_WEIGHTS_APPLIED_KEY,
)
from .store import (
    delete_prefers_edges,
    load_preference_state,
    upsert_preference_node,
    write_prefers_edges,
)
from .weights import effective_weight

logger = get_logger("user_preferences.update")


class PreferenceUpdateResult(BaseModel):
    """Outcome of one preference update run for a (user, dataset) pair."""

    status: str
    turns_applied: int = 0
    edges_written: int = 0
    edges_pruned: int = 0
    text_lines_added: int = 0


@dataclass(frozen=True, slots=True)
class PreferenceUpdateScope:
    """Resolved identity for one preference update run."""

    user: User
    dataset: Dataset

    @property
    def user_id(self) -> str:
        return str(self.user.id)

    @property
    def dataset_id(self) -> str:
        return str(self.dataset.id)


async def resolve_preference_scope(
    *,
    dataset: Union[str, UUID],
    user: Optional[User],
) -> PreferenceUpdateScope:
    resolved_user = user if user is not None else session_user.get()
    if resolved_user is None or getattr(resolved_user, "id", None) is None:
        resolved_user = await get_default_user()

    if dataset is None:
        raise CogneeValidationError(
            message="dataset is required so preference edges land in the graph they refer to.",
            name="UserPreferenceUpdateError",
        )

    writable_datasets = await get_authorized_existing_datasets([dataset], "write", resolved_user)
    if not writable_datasets:
        raise CogneeValidationError(
            message=f"Dataset '{dataset}' not found or not writable for this user.",
            name="UserPreferenceUpdateError",
        )

    return PreferenceUpdateScope(user=resolved_user, dataset=writable_datasets[0])


def _qa_metadata(row: dict) -> dict:
    metadata = row.get("memify_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _valid_rating(value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if 1 <= value <= 5 else None


def build_rating_map(feedback_rows: List[dict]) -> Dict[str, int]:
    """Map qa_id -> inferred 1-5 rating from feedback entries, latest entry winning."""
    ratings: Dict[str, int] = {}
    for row in feedback_rows:
        rating = _valid_rating(row.get("referenced_qa_rating"))
        if rating is None:
            continue
        for qa_id in row.get("referenced_qa_ids") or []:
            if isinstance(qa_id, str) and qa_id:
                ratings[qa_id] = rating
    return ratings


def resolve_turn_rating(qa_row: dict, rating_map: Dict[str, int]) -> Optional[int]:
    """Explicit human feedback wins where it exists; otherwise the inferred rating."""
    explicit = _valid_rating(qa_row.get("feedback_score"))
    if explicit is not None:
        return explicit
    qa_id = qa_row.get("qa_id")
    return rating_map.get(str(qa_id)) if qa_id else None


def extract_node_ids(used_graph_element_ids: Any) -> List[str]:
    """Node ids a turn's answer was built from. edge_ids are ignored (rule 5)."""
    if not isinstance(used_graph_element_ids, dict):
        return []
    values = used_graph_element_ids.get("node_ids")
    if not isinstance(values, list):
        return []
    return sorted({value for value in values if isinstance(value, str) and value})


def _truncate_to_whole_lines(text: str, max_chars: int) -> str:
    """Cap ``text`` at ``max_chars`` without ever cutting a line mid-sentence.

    Drops whole trailing lines until the text fits; a half instruction is worse
    than a missing one. Falls back to a plain slice only in the degenerate case
    of a single line longer than the cap.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    cut = truncated.rfind("\n")
    return truncated[:cut] if cut != -1 else truncated


def refresh_preference_text(
    existing_text: str,
    text_watermark: str,
    context_entries: List[SessionContextEntry],
) -> Tuple[str, str, int]:
    """Fold new gated preference lines into the node text; returns (text, watermark, added).

    Deterministic, no LLM call, and no content deduplication — a preference the
    user states three times is three pieces of evidence. Idempotency comes from
    the watermark (row identity, not content): only entries strictly newer than
    it are folded, newest first, and the whole string is then truncated to
    ``MAX_PREFERENCE_TEXT_CHARS`` on line boundaries so the oldest lines are
    the ones that fade, and never as half a sentence.

    Only the ``preferences`` section is used: ``goals`` is session-scoped,
    ``lessons_learned`` is promoted into the graph by distillation, and
    ``rules`` mixes durable instructions with session-bound ones.
    """
    gated = [
        entry
        for entry in context_entries
        if entry.section == ContextSection.PREFERENCES.value
        and is_context_entry_usable(entry)
        and (entry.created_at or "") > text_watermark
    ]
    if not gated:
        return existing_text, text_watermark, 0

    gated.sort(key=lambda entry: entry.created_at, reverse=True)
    new_lines = [entry.content for entry in gated]

    combined = "\n".join(new_lines)
    if existing_text:
        combined = combined + "\n" + existing_text
    combined = _truncate_to_whole_lines(combined, MAX_PREFERENCE_TEXT_CHARS)

    new_watermark = max(entry.created_at for entry in gated)
    return combined, new_watermark, len(new_lines)


async def _load_session_rows(
    session_manager,
    scope: PreferenceUpdateScope,
    session_ids: List[str],
) -> Tuple[List[Tuple[str, dict]], List[dict], List[Any]]:
    """Load QA turns (oldest first, across sessions), feedback rows, and context rows."""
    qa_rows: List[Tuple[str, dict]] = []
    feedback_rows: List[dict] = []
    context_rows: List[Any] = []

    for session_id in session_ids:
        raw_qa = await session_manager.get_session(
            user_id=scope.user_id,
            session_id=session_id,
            formatted=False,
        )
        for entry in raw_qa if isinstance(raw_qa, list) else []:
            row = entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
            qa_rows.append((session_id, row))

        raw_context = await session_manager.get_session_context_entries(
            user_id=scope.user_id,
            session_id=session_id,
        )
        for raw in raw_context or []:
            row = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {})
            if row.get("kind", "context") == "feedback":
                feedback_rows.append(row)
            else:
                context_rows.append(raw)

    # Oldest first: the update is order-sensitive by design, so recent evidence wins.
    qa_rows.sort(key=lambda item: item[1].get("time") or "")
    feedback_rows.sort(key=lambda row: row.get("created_at") or "")
    return qa_rows, feedback_rows, context_rows


async def _run_preference_update(
    scope: PreferenceUpdateScope,
    session_ids: List[str],
) -> PreferenceUpdateResult:
    config = get_base_config()
    alpha = config.preference_alpha
    beta = config.preference_beta

    session_manager = get_session_manager()
    qa_rows, feedback_rows, context_rows = await _load_session_rows(
        session_manager, scope, session_ids
    )
    rating_map = build_rating_map(feedback_rows)

    node, stored = await load_preference_state(scope.user_id, scope.dataset_id)
    turn_counter = int(node.get("turn_counter") or 0) if node else 0
    existing_text = str(node.get("text") or "") if node else ""
    text_watermark = str(node.get("text_watermark") or "") if node else ""

    # Step 1: advance the clock once for every not-yet-counted turn, rated or not.
    new_turns = [
        (session_id, row)
        for session_id, row in qa_rows
        if row.get("qa_id") and not _qa_metadata(row).get(PREFERENCE_TURN_COUNTED_KEY)
    ]
    turn_counter += len(new_turns)
    counter_moved = bool(new_turns)

    # Steps 2-4: resolve a rating per turn and move weights toward its target.
    weights_state = {target_id: dict(props) for target_id, props in stored.items()}
    pending_writes: Dict[str, float] = {}
    applied_turns: List[Tuple[str, dict]] = []
    unapplied_turns: List[Tuple[str, dict]] = []

    for session_id, row in qa_rows:
        if not row.get("qa_id"):
            continue
        if _qa_metadata(row).get(PREFERENCE_WEIGHTS_APPLIED_KEY) is True:
            continue
        rating = resolve_turn_rating(row, rating_map)
        # 3 is a no-op by construction: normalize_feedback_score(3) is exactly
        # neutral, and a convex step toward neutral would erase evidence.
        if rating is None or rating == 3:
            continue

        node_ids = extract_node_ids(row.get("used_graph_element_ids"))
        if not node_ids:
            # A usable rating with nothing to credit: mark False for a free
            # retry next run, exactly as apply_feedback_weights does.
            unapplied_turns.append((session_id, row))
            continue

        target = normalize_feedback_score(rating)
        for target_id in node_ids:
            state = weights_state.get(target_id)
            if state is None:
                current = NEUTRAL_WEIGHT
            else:
                current = effective_weight(
                    state.get("weight", NEUTRAL_WEIGHT),
                    state.get("updated_at_turn", 0),
                    turn_counter,
                    beta,
                )
            updated = stream_update_weight(current, target, alpha)
            weights_state[target_id] = {"weight": updated, "updated_at_turn": turn_counter}
            pending_writes[target_id] = updated
        applied_turns.append((session_id, row))

    # Step 7 (computed before persisting so the node is written exactly once).
    text, new_watermark, text_lines_added = refresh_preference_text(
        existing_text,
        text_watermark,
        coerce_active_context_entries(context_rows),
    )

    # Persist the node. It is only created when there is something to store —
    # a session of turns with no ratings and no stated preferences leaves no
    # node behind. Losing those early clock ticks is harmless: with no edges,
    # there is nothing to decay yet.
    node_dirty = (
        bool(pending_writes) or text_lines_added > 0 or (node is not None and counter_moved)
    )
    if node_dirty:
        await upsert_preference_node(
            scope.user_id,
            scope.dataset_id,
            text=text,
            turn_counter=turn_counter,
            text_watermark=new_watermark,
        )

    # Step 4 write: weight and updated_at_turn always land together.
    write_ok = True
    edges_written = 0
    if pending_writes:
        try:
            await write_prefers_edges(scope.user_id, scope.dataset_id, pending_writes, turn_counter)
            edges_written = len(pending_writes)
        except Exception as error:
            logger.warning("Preference update: prefers-edge write failed: %s", error)
            write_ok = False

    # Step 6: prune whenever the counter moved — the clock advances on unrated
    # turns too, so an edge can decay out of existence in a stretch of
    # conversation that produced no ratings at all.
    edges_pruned = 0
    if counter_moved:
        state_for_prune = (
            weights_state
            if write_ok
            else {target_id: dict(props) for target_id, props in stored.items()}
        )
        # Rows already at exactly neutral are skipped: on backends without
        # delete_edge_triples the prune fallback rewrites an edge as neutral
        # instead of deleting it, and re-collecting that tombstone here would
        # rewrite it (and count it in edges_pruned) on every later run.
        prune_ids = [
            target_id
            for target_id, state in state_for_prune.items()
            if state.get("weight", NEUTRAL_WEIGHT) != NEUTRAL_WEIGHT
            and abs(
                effective_weight(
                    state.get("weight", NEUTRAL_WEIGHT),
                    state.get("updated_at_turn", 0),
                    turn_counter,
                    beta,
                )
                - NEUTRAL_WEIGHT
            )
            <= PREFERENCE_DELETE_THRESHOLD
        ]
        if prune_ids:
            await delete_prefers_edges(scope.user_id, scope.dataset_id, prune_ids)
            edges_pruned = len(prune_ids)

    # Step 5: mark processed. update_qa MERGES the metadata dict into the
    # stored one (the cache adapters overlay keys), so pass ONLY this stage's
    # keys — copying the whole row snapshot could write another stage's stale
    # value over a fresher one. Both values are real bools (the field validates
    # as Dict[str, bool]); and the applied key records the actual outcome, not
    # a hard True — the skip guard is `is True`, so False gives a free retry.
    metadata_updates: Dict[Tuple[str, str], dict] = {}
    for session_id, row in new_turns:
        metadata_updates[(session_id, row["qa_id"])] = {PREFERENCE_TURN_COUNTED_KEY: True}
    for session_id, row in applied_turns:
        metadata = metadata_updates.setdefault((session_id, row["qa_id"]), {})
        metadata[PREFERENCE_WEIGHTS_APPLIED_KEY] = bool(write_ok)
    for session_id, row in unapplied_turns:
        metadata = metadata_updates.setdefault((session_id, row["qa_id"]), {})
        metadata[PREFERENCE_WEIGHTS_APPLIED_KEY] = False

    for (session_id, qa_id), metadata in metadata_updates.items():
        await session_manager.update_qa(
            user_id=scope.user_id,
            session_id=session_id,
            qa_id=qa_id,
            memify_metadata=metadata,
        )

    changed = counter_moved or edges_written or edges_pruned or text_lines_added
    return PreferenceUpdateResult(
        status="completed" if changed else "no_changes",
        turns_applied=len(applied_turns) if write_ok else 0,
        edges_written=edges_written,
        edges_pruned=edges_pruned,
        text_lines_added=text_lines_added,
    )


async def update_user_preferences(
    session_ids: List[str],
    dataset: Union[str, UUID],
    user: Optional[User] = None,
) -> PreferenceUpdateResult:
    """Fold rated turns and stated preferences from ``session_ids`` into the
    caller's preference subgraph for ``dataset``.

    Re-running is safe: each turn is counted once and its rating spent once
    (``memify_metadata`` markers), and each text row is folded once (watermark).

    Gated on ``PERSONALIZATION_ENABLED`` here — the top of the writing path —
    so every caller is covered: with the flag off (the default) nothing is
    written anywhere (no preference node, no ``prefers`` edges, no
    ``memify_metadata`` keys) and a default deployment stays byte-identical
    to today.
    """
    if not get_base_config().personalization_enabled:
        return PreferenceUpdateResult(status="personalization_disabled")

    if not session_ids:
        return PreferenceUpdateResult(status="no_changes")

    scope = await resolve_preference_scope(dataset=dataset, user=user)
    async with set_database_global_context_variables(scope.dataset.id, scope.dataset.owner_id):
        return await _run_preference_update(scope, list(session_ids))
