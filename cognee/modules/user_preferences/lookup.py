"""The memoized retrieval-time read of the active user's preferences.

One call returns everything the retrieval path needs: the render-ready
preference text for the guidance channel, and the already-decayed ``prefers``
weight map for ranking. It fails open everywhere (hard rule 6): flag off,
missing identity, no node, or any error returns ``("", {})`` at debug-log
cost — a missing or broken preference node must never fail a search.
"""

from contextvars import ContextVar
from typing import Dict, Optional, Tuple

from cognee.base_config import get_base_config
from cognee.context_global_variables import current_dataset_id, session_user
from cognee.shared.logging_utils import get_logger

from .constants import NEUTRAL_WEIGHT, PREFERENCE_RENDER_HEADER
from .store import load_preference_state
from .weights import effective_weight

logger = get_logger("user_preferences.lookup")

# Memoized per (user_id, dataset_id) so both retrieval lanes of a concurrent
# session turn and the completion call share one graph read. A ContextVar
# rather than retriever state because the completion-side read happens in
# infrastructure/session, where no retriever is in scope.
_active_preferences_cache: ContextVar[
    Optional[Tuple[Tuple[str, str], Tuple[str, Dict[str, float]]]]
] = ContextVar("active_preferences_cache", default=None)


async def load_active_preferences() -> Tuple[str, Dict[str, float]]:
    """Load the active user's preference text and decayed prefers weights.

    Returns ``(preference_text, weights)`` for the (``session_user``,
    ``current_dataset_id``) pair in context. The text arrives with the render
    header already prepended (empty text stays ``""`` with no stray header) —
    this is the single funnel to every render site, so no caller composes
    that string itself. Weights arrive with decay already applied via
    ``effective_weight``, so callers never see a raw stored value.
    """
    try:
        config = get_base_config()
        if not config.personalization_enabled:
            return "", {}

        user_id = getattr(session_user.get(), "id", None)
        dataset_id = current_dataset_id.get()
        if user_id is None or dataset_id is None:
            return "", {}

        cache_key = (str(user_id), str(dataset_id))
        cached = _active_preferences_cache.get()
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        node, stored = await load_preference_state(cache_key[0], cache_key[1])
        if node is None:
            result: Tuple[str, Dict[str, float]] = ("", {})
        else:
            turn_counter = int(node.get("turn_counter", 0) or 0)
            weights = {
                target_id: effective_weight(
                    entry.get("weight", NEUTRAL_WEIGHT),
                    entry.get("updated_at_turn", 0),
                    turn_counter,
                    config.preference_beta,
                )
                for target_id, entry in stored.items()
            }
            text = str(node.get("text") or "")
            rendered = PREFERENCE_RENDER_HEADER + "\n" + text if text.strip() else ""
            result = (rendered, weights)

        _active_preferences_cache.set((cache_key, result))
        return result
    except Exception as error:
        logger.debug("Preference lookup failed open: %s", error)
        return "", {}
