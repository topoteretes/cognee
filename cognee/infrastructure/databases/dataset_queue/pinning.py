"""Engine-cache pin predicate built on the dataset queue's slot registry."""


class DatasetQueuePinPredicate:
    """Pin predicate for ``closing_lru_cache``, bound to a named parameter.

    The predicate is True when the engine's database belongs to a dataset
    currently holding a dataset-queue slot: capacity eviction must not close
    an engine that an admitted pipeline is still using — a mid-cognify
    dataset idling on an LLM call looks least-recently-used exactly when
    closing it is most dangerous.

    It is created with the *name* of the cached factory's parameter that
    holds the database name; ``closing_lru_cache`` calls
    :meth:`bind_signature` at decoration time to resolve that name against
    the factory's real signature. An unknown name fails loudly at import
    instead of silently pinning nothing when the signature drifts.
    Per-dataset databases are named ``<dataset_id>.<ext>``, so the name stem
    maps directly to the queue's active ids.

    The signal is deliberately queue-scoped: with the dataset queue disabled
    there are no slots, nothing pins, and capacity eviction falls back to
    plain recency — the close→reopen ordering (the cache's pending-close
    registry) still applies. Queue slots are the only activity signal with a
    guaranteed expiry (scoped release plus the task-end backstop), so pins
    can never wedge the cache the way a GC-dependent signal (e.g. "a lease
    proxy is still held") could.

    Calls run under the cache lock: kept cheap, never re-enter the cache.
    """

    def __init__(self, database_name_parameter: str):
        self._parameter = database_name_parameter
        self._index = None

    def bind_signature(self, parameter_positions: dict) -> None:
        """Resolve the parameter name to its positional index in the cache key.

        Called once by ``closing_lru_cache`` with the cached function's
        name-to-position map. Raises when the parameter does not exist.
        """
        if self._parameter not in parameter_positions:
            raise ValueError(
                f"Pin predicate parameter {self._parameter!r} is not a parameter "
                f"of the cached function (has: {sorted(parameter_positions)})"
            )
        self._index = parameter_positions[self._parameter]

    def __call__(self, key) -> bool:
        from cognee.infrastructure.databases.dataset_queue import dataset_queue

        if self._index is None:
            raise RuntimeError(
                f"Pin predicate for {self._parameter!r} was never bound to a signature; "
                "use it via the closing_lru_cache(pinned_predicate=...) decorator"
            )
        database_name = key[self._index] if len(key) > self._index else ""
        if not isinstance(database_name, str) or not database_name:
            return False
        active = dataset_queue().active_dataset_ids()
        return bool(active) and database_name.split(".", 1)[0] in active


def dataset_queue_pin_predicate(database_name_parameter: str) -> DatasetQueuePinPredicate:
    """Build a pin predicate keyed on the named database-name parameter."""
    return DatasetQueuePinPredicate(database_name_parameter)
