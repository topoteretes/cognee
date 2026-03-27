"""Global memory cleanup runtime for long-lived Cognee resources."""

import asyncio
import os
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from cognee.shared.logging_utils import get_logger

logger = get_logger()

CGROUP_V2_MEMORY_MAX_PATH = Path("/sys/fs/cgroup/memory.max")
CGROUP_V1_MEMORY_LIMIT_PATH = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

DEFAULT_PRESSURE_THRESHOLD = 0.85
DEFAULT_STALE_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 30.0
DEFAULT_COOLDOWN_SECONDS = 30.0
_UNLIMITED_CGROUP_VALUES = {"", "max"}


@runtime_checkable
class MemoryItem(Protocol):
    def memory_used(self) -> int:
        """Return estimated memory usage in bytes."""

    def last_accessed_ts(self) -> float:
        """Return the last-access Unix timestamp."""

    def clean(self) -> Any:
        """Release the held resource."""


@runtime_checkable
class MemoryComponent(Protocol):
    def get_items(self) -> list[MemoryItem]:
        """Return cleanable memory items owned by the component."""


@dataclass(frozen=True)
class MemoryPressureSnapshot:
    total_used: int
    effective_limit: int
    ratio: float


def is_memory_item(obj: Any) -> bool:
    return all(hasattr(obj, attr) for attr in ("memory_used", "last_accessed_ts", "clean"))


def _read_limit_value(path: Path) -> Optional[int]:
    try:
        raw_value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if raw_value in _UNLIMITED_CGROUP_VALUES:
        return None

    try:
        value = int(raw_value)
    except ValueError:
        return None

    return value if value > 0 else None


def get_physical_memory_limit() -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * physical_pages)


def _normalize_limit(limit: Optional[int], physical_limit: int) -> Optional[int]:
    if limit is None or limit <= 0:
        return None

    if limit >= 2**60:
        return None

    return min(limit, physical_limit)


def get_cgroup_v2_memory_limit() -> Optional[int]:
    return _read_limit_value(CGROUP_V2_MEMORY_MAX_PATH)


def get_cgroup_v1_memory_limit() -> Optional[int]:
    return _read_limit_value(CGROUP_V1_MEMORY_LIMIT_PATH)


def get_effective_memory_limit() -> int:
    physical_limit = get_physical_memory_limit()

    cgroup_v2_limit = _normalize_limit(get_cgroup_v2_memory_limit(), physical_limit)
    if cgroup_v2_limit is not None:
        return cgroup_v2_limit

    cgroup_v1_limit = _normalize_limit(get_cgroup_v1_memory_limit(), physical_limit)
    if cgroup_v1_limit is not None:
        return cgroup_v1_limit

    return physical_limit


def get_process_rss(pid: Optional[int] = None) -> int:
    try:
        import psutil  # type: ignore

        process = psutil.Process(pid) if pid is not None else psutil.Process()
        return int(process.memory_info().rss)
    except Exception:
        pass

    statm_path = Path("/proc/self/statm") if pid is None else Path(f"/proc/{pid}/statm")

    try:
        rss_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return 0


def get_current_process_rss() -> int:
    return get_process_rss()


def _run_cleanup_result(result: Any) -> None:
    if not asyncio.iscoroutine(result):
        return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(result)
    except RuntimeError:
        try:
            asyncio.run(result)
        except Exception:
            logger.warning("Failed to run async resource cleanup", exc_info=True)


class MemoryCleanupManager:
    """Singleton-style manager that cleans stale or memory-heavy resources."""

    def __init__(
        self,
        *,
        pressure_threshold: float = DEFAULT_PRESSURE_THRESHOLD,
        stale_timeout_seconds: float = DEFAULT_STALE_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        now_provider: Callable[[], float] = time.time,
        current_rss_provider: Callable[[], int] = get_current_process_rss,
        effective_limit_provider: Callable[[], int] = get_effective_memory_limit,
    ):
        self._pressure_threshold = pressure_threshold
        self._stale_timeout_seconds = stale_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._cooldown_seconds = cooldown_seconds
        self._now_provider = now_provider
        self._current_rss_provider = current_rss_provider
        self._effective_limit_provider = effective_limit_provider

        self._components: weakref.WeakSet[Any] = weakref.WeakSet()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_pressure_cleanup_ts = 0.0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="cognee-memory-cleanup",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, reset: bool = False) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop_event.set()

        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._poll_interval_seconds + 1.0))

        if reset:
            with self._lock:
                self._components = weakref.WeakSet()
                self._last_pressure_cleanup_ts = 0.0

    def register_component(self, component: MemoryComponent) -> None:
        with self._lock:
            self._components.add(component)
        self.start()

    def unregister_component(self, component: MemoryComponent) -> None:
        with self._lock:
            self._components.discard(component)

    def get_pressure_snapshot(
        self, items: Optional[list[MemoryItem]] = None
    ) -> MemoryPressureSnapshot:
        if items is None:
            items = self._collect_items()

        extra_item_memory = 0
        for item in items:
            extra_item_memory += max(0, self._safe_memory_used(item))

        current_rss = max(0, self._current_rss_provider())
        effective_limit = max(1, self._effective_limit_provider())
        total_used = current_rss + extra_item_memory
        return MemoryPressureSnapshot(
            total_used=total_used,
            effective_limit=effective_limit,
            ratio=total_used / effective_limit,
        )

    def run_cleanup_cycle(self) -> dict[str, float | int]:
        now = self._now_provider()
        stats: dict[str, float | int] = {
            "considered": 0,
            "stale_cleaned": 0,
            "pressure_cleaned": 0,
            "ratio_before": 0.0,
            "ratio_after": 0.0,
        }

        initial_items = self._collect_items()
        stats["considered"] = len(initial_items)

        stale_items = sorted(
            [item for item in initial_items if self._is_stale(item, now)],
            key=self._safe_last_accessed_ts,
        )
        stats["stale_cleaned"] = self._clean_items(stale_items)

        pressure_snapshot = self.get_pressure_snapshot(self._collect_items())
        stats["ratio_before"] = pressure_snapshot.ratio

        cooldown_elapsed = (now - self._last_pressure_cleanup_ts) >= self._cooldown_seconds
        if pressure_snapshot.ratio >= self._pressure_threshold and cooldown_elapsed:
            pressure_cleaned = 0
            for item in sorted(self._collect_items(), key=self._safe_last_accessed_ts):
                current_snapshot = self.get_pressure_snapshot(self._collect_items())
                if current_snapshot.ratio < self._pressure_threshold:
                    break

                if self._clean_item(item):
                    pressure_cleaned += 1

            if pressure_cleaned > 0:
                self._last_pressure_cleanup_ts = now
            stats["pressure_cleaned"] = pressure_cleaned

        final_snapshot = self.get_pressure_snapshot(self._collect_items())
        stats["ratio_after"] = final_snapshot.ratio

        cleaned_count = int(stats["stale_cleaned"]) + int(stats["pressure_cleaned"])
        if cleaned_count > 0:
            logger.info(
                "Memory cleanup cycle cleaned %s items (stale=%s pressure=%s ratio_before=%.3f ratio_after=%.3f)",
                cleaned_count,
                stats["stale_cleaned"],
                stats["pressure_cleaned"],
                stats["ratio_before"],
                stats["ratio_after"],
            )

        return stats

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            try:
                self.run_cleanup_cycle()
            except Exception:
                logger.warning("Memory cleanup cycle failed", exc_info=True)

    def _collect_items(self) -> list[MemoryItem]:
        with self._lock:
            components = list(self._components)

        items: list[MemoryItem] = []
        for component in components:
            try:
                component_items = component.get_items()
            except Exception:
                logger.warning(
                    "Failed to collect memory items from %s",
                    type(component).__name__,
                    exc_info=True,
                )
                continue

            for item in component_items:
                if is_memory_item(item):
                    items.append(item)

        return items

    def _is_stale(self, item: MemoryItem, now: float) -> bool:
        return (now - self._safe_last_accessed_ts(item)) >= self._stale_timeout_seconds

    def _clean_items(self, items: list[MemoryItem]) -> int:
        cleaned = 0
        for item in items:
            if self._clean_item(item):
                cleaned += 1
        return cleaned

    def _clean_item(self, item: MemoryItem) -> bool:
        try:
            _run_cleanup_result(item.clean())
            return True
        except Exception:
            logger.warning("Failed to clean memory item %s", type(item).__name__, exc_info=True)
            return False

    @staticmethod
    def _safe_memory_used(item: MemoryItem) -> int:
        try:
            return int(item.memory_used())
        except Exception:
            logger.warning(
                "Failed to read memory usage from %s",
                type(item).__name__,
                exc_info=True,
            )
            return 0

    @staticmethod
    def _safe_last_accessed_ts(item: MemoryItem) -> float:
        try:
            return float(item.last_accessed_ts())
        except Exception:
            logger.warning(
                "Failed to read access timestamp from %s",
                type(item).__name__,
                exc_info=True,
            )
            return 0.0


_manager_lock = threading.Lock()
_manager: Optional[MemoryCleanupManager] = None


def get_memory_cleanup_manager() -> MemoryCleanupManager:
    global _manager

    with _manager_lock:
        if _manager is None:
            _manager = MemoryCleanupManager()
        return _manager


def stop_memory_cleanup_manager(*, reset: bool = False) -> None:
    global _manager

    with _manager_lock:
        manager = _manager

    if manager is None:
        return

    manager.stop(reset=reset)

    if reset:
        with _manager_lock:
            _manager = None
