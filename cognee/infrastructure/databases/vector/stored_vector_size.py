"""Pick one width for a store whose collections do not all agree on one.

Every collection in a dataset is created with the width of the engine bound at
the time (``create_collection``), so a store built entirely after cognee started
recording the embedding model holds exactly one width. A store that predates it
can hold two: change ``EMBEDDING_MODEL`` on a live deployment and the existing
collections keep the old width (writes to them fail), while the next new node
type creates its collection at the new one.

``get_stored_vector_size`` has to answer with a single number for such a store,
and for a legacy registry row that number is then recorded permanently. Taking
whichever collection the store happens to list first makes that an arbitrary
choice between "the check refuses a config that half the store accepts" and
"the check passes a config that half the store rejects". This makes the choice
explicit — the width most collections use, smallest on a tie — and says so in
the log, because no single answer is right for a store in that state.
"""

from collections import Counter
from collections.abc import Iterable

from cognee.shared.logging_utils import get_logger

logger = get_logger("stored_vector_size")


def choose_stored_vector_size(widths: Iterable[int], store: str) -> int | None:
    """The width to report for a store, warning when its collections disagree.

    Args:
        widths: Every collection's vector width, in any order.
        store: What to call this store in the warning (the adapter's name).
    """
    counts = Counter(widths)
    if not counts:
        return None

    # Most collections first, then the smaller width — a stable answer for the
    # same store rather than one that depends on listing order.
    chosen = min(counts, key=lambda width: (-counts[width], width))

    if len(counts) > 1:
        logger.warning(
            "%s holds collections with different vector widths %s; reporting %d (used by "
            "%d of %d collections). A store gets into this state when EMBEDDING_MODEL "
            "changes while it already holds vectors: the older collections keep the old "
            "width and reject every write. Re-embed this dataset to make it consistent.",
            store,
            dict(sorted(counts.items())),
            chosen,
            counts[chosen],
            sum(counts.values()),
        )

    return chosen
