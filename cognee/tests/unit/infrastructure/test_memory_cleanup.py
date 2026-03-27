import pytest

from cognee.infrastructure.memory_cleanup import (
    MemoryCleanupManager,
    get_effective_memory_limit,
)


class _FakeItem:
    def __init__(self, memory_used: int, last_accessed: float):
        self._memory_used = memory_used
        self._last_accessed = last_accessed
        self.cleaned = False

    def memory_used(self) -> int:
        return 0 if self.cleaned else self._memory_used

    def last_accessed_ts(self) -> float:
        return self._last_accessed

    def clean(self) -> None:
        self.cleaned = True


class _FakeComponent:
    def __init__(self, *items):
        self._items = list(items)

    def get_items(self):
        return list(self._items)


def test_effective_limit_prefers_cgroup_v2(monkeypatch):
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_physical_memory_limit",
        lambda: 1_000,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_cgroup_v2_memory_limit",
        lambda: 750,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_cgroup_v1_memory_limit",
        lambda: 500,
    )

    assert get_effective_memory_limit() == 750


def test_effective_limit_falls_back_to_cgroup_v1(monkeypatch):
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_physical_memory_limit",
        lambda: 1_000,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_cgroup_v2_memory_limit",
        lambda: None,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_cgroup_v1_memory_limit",
        lambda: 650,
    )

    assert get_effective_memory_limit() == 650


def test_effective_limit_falls_back_to_physical_memory(monkeypatch):
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_physical_memory_limit",
        lambda: 1_000,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_cgroup_v2_memory_limit",
        lambda: None,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.memory_cleanup.get_cgroup_v1_memory_limit",
        lambda: None,
    )

    assert get_effective_memory_limit() == 1_000


def test_stale_cleanup_only_cleans_idle_items():
    manager = MemoryCleanupManager(
        stale_timeout_seconds=10,
        poll_interval_seconds=60,
        cooldown_seconds=0,
        now_provider=lambda: 100,
        current_rss_provider=lambda: 10,
        effective_limit_provider=lambda: 1_000,
    )
    stale_item = _FakeItem(memory_used=50, last_accessed=10)
    fresh_item = _FakeItem(memory_used=50, last_accessed=95)
    component = _FakeComponent(stale_item, fresh_item)
    manager.register_component(component)

    try:
        stats = manager.run_cleanup_cycle()
    finally:
        manager.stop(reset=True)

    assert stats["stale_cleaned"] == 1
    assert stale_item.cleaned is True
    assert fresh_item.cleaned is False


def test_pressure_cleanup_prefers_oldest_items_first():
    manager = MemoryCleanupManager(
        pressure_threshold=0.85,
        stale_timeout_seconds=1_000,
        poll_interval_seconds=60,
        cooldown_seconds=0,
        now_provider=lambda: 100,
        current_rss_provider=lambda: 40,
        effective_limit_provider=lambda: 100,
    )
    oldest = _FakeItem(memory_used=30, last_accessed=10)
    newest = _FakeItem(memory_used=30, last_accessed=90)
    component = _FakeComponent(oldest, newest)
    manager.register_component(component)

    try:
        stats = manager.run_cleanup_cycle()
    finally:
        manager.stop(reset=True)

    assert stats["pressure_cleaned"] == 1
    assert oldest.cleaned is True
    assert newest.cleaned is False
