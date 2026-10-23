"""Local telemetry resource bounds; content/privacy schemas belong to SDK-1167."""

import json
import math
import os
from dataclasses import dataclass

# Deliberately closed: adding a cloud event requires a volume-policy review.
CLOUD_EVENTS = frozenset(
    {
        "Pipeline Run Started",
        "Pipeline Run Completed",
        "Pipeline Run Errored",
        "Incremental Update Run Started",
        "Incremental Update Run Completed",
        "cognee.cognify DEFAULT TASKS CREATION ERRORED",
        "cognee.remember",
        "cognee.remember.code_graph",
        "cognee.remember.import",
        "cognee.recall",
        "cognee.recall ERRORED",
        "cognee.improve",
        "cognee.forget",
        "cognee.push",
        "cognee.export",
    }
    | {
        f"cognee.{operation} EXECUTION {outcome}"
        for operation in (
            "add",
            "cognify",
            "memify",
            "search",
            "remember",
            "recall",
            "improve",
            "forget",
            "delete",
            "update",
            "push",
            "export",
        )
        for outcome in ("STARTED", "COMPLETED", "ERRORED")
    }
)


_MAX_LIMITS = {
    "event_bytes": 64 * 1024,
    "queue_bytes": 16 * 1024 * 1024,
    "batch_bytes": 256 * 1024,
    "events_per_minute": 1_000_000,
    "bytes_per_minute": 2**30,
    "requests_per_minute": 6_000,
}


@dataclass(frozen=True)
class TelemetryLimits:
    """Finite per-process limits, read once at sender construction."""

    event_bytes: int = 16 * 1024
    queue_bytes: int = 1024 * 1024
    batch_bytes: int = 64 * 1024
    events_per_minute: int = 120
    bytes_per_minute: int = 256 * 1024
    requests_per_minute: int = 60

    def __post_init__(self):
        for field, value in vars(self).items():
            if type(value) is not int or not 1 <= value <= _MAX_LIMITS[field]:
                raise ValueError("Telemetry limit is outside the supported finite range")
        if self.event_bytes + 2 > self.batch_bytes or self.event_bytes > self.queue_bytes:
            raise ValueError("Telemetry event must fit in a batch and queue")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for field, default in vars(defaults).items():
            try:
                value = int(os.getenv(f"TELEMETRY_{field.upper()}", str(default)))
            except ValueError:
                value = default
            values[field] = value if 1 <= value <= _MAX_LIMITS[field] else default
        # Invalid cross-field settings fall back together; never become unbounded.
        try:
            return cls(**values)
        except ValueError:
            return defaults


class PayloadTooLarge(ValueError):
    """The payload exceeds the byte, nesting, or traversal budget."""


def bounded_json(value, limit: int) -> bytes:
    """Encode within byte/traversal budgets without an unbounded JSON allocation.

    At most 2,048 values / 16 levels are visited. Strings are length-checked
    before escaping; a rejected scalar's temporary representation is at most
    12 * limit + 2 bytes (two six-byte escapes per non-BMP character).
    """
    encoder = _BoundedEncoder(limit)
    encoder.encode(value)
    return bytes(encoder.output)


class _BoundedEncoder:
    # A recursive closure creates a reference cycle retaining its output until
    # cyclic GC runs. An ordinary method releases buffers immediately, even
    # when the embedding application disables cyclic GC.
    __slots__ = ("limit", "nodes", "output")

    def __init__(self, limit):
        self.output = bytearray()
        self.limit = limit
        self.nodes = 0

    def append(self, part: bytes):
        if len(self.output) + len(part) > self.limit:
            raise PayloadTooLarge("Telemetry payload exceeds byte budget")
        self.output.extend(part)

    def encode(self, item, depth=0):
        self.nodes += 1
        if depth > 16 or self.nodes > 2048:
            raise PayloadTooLarge("Telemetry payload exceeds traversal budget")
        kind = type(item)
        if kind in (dict, list, tuple):
            self.append(b"{" if kind is dict else b"[")
            entries = item.items() if kind is dict else enumerate(item)
            for index, (key, child) in enumerate(entries):
                if index:
                    self.append(b",")
                if kind is dict:
                    if type(key) is not str:
                        raise ValueError("Telemetry keys must be strings")
                    self.encode(key, depth + 1)
                    self.append(b":")
                self.encode(child, depth + 1)
            self.append(b"}" if kind is dict else b"]")
        else:
            if kind is str:
                if len(item) > self.limit:
                    raise PayloadTooLarge("Telemetry string exceeds byte budget")
            elif kind is int:
                if item.bit_length() > 64:
                    raise ValueError("Telemetry integer exceeds 64 bits")
            elif kind is float:
                if not math.isfinite(item):
                    raise ValueError("Telemetry number must be finite")
            elif item is not None and kind is not bool:
                raise ValueError("Telemetry value is not a JSON primitive")
            self.append(json.dumps(item, ensure_ascii=True, allow_nan=False).encode("ascii"))


class TokenBucket:
    """One minute of burst capacity, refilled continuously using a monotonic clock."""

    def __init__(self, capacity: int, now: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = now

    def available(self, amount: int, now: float) -> bool:
        self.tokens = min(
            self.capacity, self.tokens + max(0, now - self.updated) * self.capacity / 60
        )
        self.updated = now
        return amount <= self.tokens

    def take(self, amount: int):
        self.tokens -= amount
