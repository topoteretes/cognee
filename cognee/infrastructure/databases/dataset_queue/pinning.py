"""Engine-cache pin predicate built on the dataset queue's slot registry."""


def dataset_queue_pin_predicate(database_name_index: int):
    """Build a ``pinned_predicate`` for ``closing_lru_cache``.

    The predicate is True when the engine's database belongs to a dataset
    currently holding a dataset-queue slot: capacity eviction must not close
    an engine that an admitted pipeline is still using — a mid-cognify
    dataset idling on an LLM call looks least-recently-used exactly when
    closing it is most dangerous.

    ``database_name_index`` locates the database name in the engine cache
    key; per-dataset databases are named ``<dataset_id>.<ext>``, so the name
    stem maps directly to the queue's active ids. The predicate runs under
    the cache lock: it stays cheap and never re-enters the cache.
    """

    def is_pinned(key) -> bool:
        from cognee.infrastructure.databases.dataset_queue import dataset_queue

        database_name = key[database_name_index] if len(key) > database_name_index else ""
        if not isinstance(database_name, str) or not database_name:
            return False
        active = dataset_queue().active_dataset_ids()
        return bool(active) and database_name.split(".", 1)[0] in active

    return is_pinned
