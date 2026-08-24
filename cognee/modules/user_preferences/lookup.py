"""The memoized retrieval-time read of the active user's preferences.

One graph read per context backs everything the retrieval path needs,
exposed as three views over the same cached state: the render-ready
preference text for standalone guidance sites, the raw lines for the session
guidance block (which owns rendering and sizing there), and the
already-decayed ``prefers`` weight map for ranking. Sharing that read across
an ``asyncio.gather`` fan-out requires ``warm_preference_cache`` in the
parent first — see the cache comment below. It fails open everywhere (hard
rule 6): flag off, missing identity, no node, or any error returns the empty
value at debug-log cost — a missing or broken preference node must never
fail a search.
"""

from contextvars import ContextVar
from typing import Dict, List, Optional, Tuple

from cognee.base_config import get_base_config
from cognee.context_global_variables import current_dataset_id, session_user
from cognee.shared.logging_utils import get_logger

from .constants import NEUTRAL_WEIGHT, PREFERENCE_RENDER_HEADER
from .store import load_preference_state
from .weights import effective_weight

logger = get_logger("user_preferences.lookup")

# Memoized per (user_id, dataset_id). A ContextVar rather than retriever state
# because the completion-side read happens in infrastructure/session, where no
# retriever is in scope. Propagation caveat: a value set inside an asyncio task
# (e.g. one lane of an ``asyncio.gather``) is invisible to its sibling tasks
# and to the parent — only plain ``await``s share the caller's context. So the
# cache is shared across a fan-out only when it is warmed in the parent first
# (tasks copy the parent context at creation); that is ``warm_preference_cache``,
# called before the lane fan-out in ``session_aware_completion``. A read that
# happens inside one lane without warming is NOT free for the other lane.
_active_preferences_cache: ContextVar[
    Optional[Tuple[Tuple[str, str], Tuple[str, Dict[str, float]]]]
] = ContextVar("active_preferences_cache", default=None)


async def _load_raw_preferences() -> Tuple[str, Dict[str, float]]:
    """Load and memoize the raw node text and decayed weights; fail-open -> ("", {})."""
    try:
        config = get_base_config()
        if not config.personalization_enabled:
            logger.debug("Preference lookup skipped: PERSONALIZATION_ENABLED is off")
            return "", {}

        user_id = getattr(session_user.get(), "id", None)
        dataset_id = current_dataset_id.get()
        if user_id is None or dataset_id is None:
            # dataset_id is only set when exactly one dataset resolves, so a
            # search spanning several datasets never personalizes.
            logger.debug(
                "Preference lookup skipped: no %s in context",
                "user" if user_id is None else "single dataset",
            )
            return "", {}

        cache_key = (str(user_id), str(dataset_id))
        cached = _active_preferences_cache.get()
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        node, stored = await load_preference_state(cache_key[0], cache_key[1])
        if node is None:
            logger.debug(
                "Preference lookup found no preference node yet for user=%s dataset=%s",
                cache_key[0],
                cache_key[1],
            )
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
            result = (str(node.get("text") or ""), weights)

        _active_preferences_cache.set((cache_key, result))
        return result
    except Exception as error:
        logger.debug("Preference lookup failed open: %s", error)
        return "", {}


async def warm_preference_cache() -> None:
    """Populate the memoized preference read in the *current* context.

    Call this in the parent context before fanning out asyncio tasks: tasks
    copy the context at creation, so a cache warmed here is inherited by every
    lane and still visible to the parent afterwards, while a read performed
    inside one lane is invisible to its siblings and the parent. Fail-open and
    a cheap no-op when the flag is off or no identity is in context.
    """
    await _load_raw_preferences()


async def load_preference_text() -> str:
    """Load the active user's preference text for the guidance channel.

    Returns the text for the (``session_user``, ``current_dataset_id``) pair
    in context, with the render header already prepended (empty text stays
    ``""`` with no stray header) — this is the single funnel to every
    standalone render site, so no caller composes that string itself.
    """
    text, _weights = await _load_raw_preferences()
    return PREFERENCE_RENDER_HEADER + "\n" + text if text.strip() else ""


async def load_preference_weights() -> Dict[str, float]:
    """Load the active user's decayed prefers weight map for ranking.

    Returns the weights for the (``session_user``, ``current_dataset_id``)
    pair in context, with decay already applied via ``effective_weight``, so
    callers never see a raw stored value. Shares the memoized graph read with
    ``load_preference_text``.
    """
    _text, weights = await _load_raw_preferences()
    return weights


async def load_active_preference_lines() -> List[str]:
    """The node's stated-preference lines, newest first, unrendered.

    For callers that feed the lines into an existing guidance block (the
    session-context builder's ``Preferences`` section) instead of rendering a
    standalone preference block — the block owner does the sizing and heading.
    """
    text, _weights = await _load_raw_preferences()
    return [line.strip() for line in text.splitlines() if line.strip()]
