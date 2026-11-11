"""Opt-in, payload-free storage timing; failures in telemetry cannot fail a write."""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

from cognee.shared.logging_utils import get_logger

logger = get_logger("storage_trace")
_batch: ContextVar[dict | None] = ContextVar("storage_trace_batch", default=None)
_sink: ContextVar[object | None] = ContextVar("storage_trace_sink", default=None)


@contextmanager
def collect_storage_events(sink):
    """Collect timing dicts in a benchmark without logging input text or vectors."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


@contextmanager
def storage_batch(collection: str, rows: int):
    enabled = (
        _sink.get() is not None or os.getenv("COGNEE_STORAGE_TRACE", "false").lower() == "true"
    )
    if not enabled:
        yield
        return
    token = _batch.set({"batch_id": uuid4().hex, "collection": collection, "rows": rows})
    try:
        yield
    finally:
        _batch.reset(token)


@contextmanager
def storage_span(stage: str):
    batch = _batch.get()
    if batch is None:
        yield
        return
    start = perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        try:
            event = {
                **batch,
                "stage": stage,
                "start_s": start,
                "elapsed_s": perf_counter() - start,
                "ok": ok,
            }
            sink = _sink.get()
            if sink is not None:
                sink(event)
            else:
                logger.info("COGNEE_STORAGE_TIMING %s", json.dumps(event))
        except Exception:
            logger.debug("Ignoring storage timing failure", exc_info=True)
