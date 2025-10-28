import pytest
import threading
import time
from multiprocessing import Process

from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter


def _process_worker(process_name: str):
    """Helper function for multiprocessing tests (must be at module level for pickling)."""
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="test_two_processes")
    cache.acquire_lock()
    time.sleep(0.1)
    cache.release_lock()
    cache.cache.close()


def test_can_be_instantiated():
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="test_instantiate")
    cache.cache.close()


def test_can_acquire_lock():
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="test_acquire")
    cache.acquire_lock()
    cache.release_lock()
    cache.cache.close()


def test_can_release_lock():
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="test_release")
    cache.acquire_lock()
    cache.release_lock()
    cache.cache.close()


def test_releasing_non_acquired_lock():
    try:
        cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="test_non_acquired")
        cache.acquire_lock()
        cache.release_lock()
        cache.cache.close()
    except Exception as e:
        pytest.fail(f"Failed: {e}")


@pytest.mark.timeout(15)
def test_two_threads():
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="test_two_threads")
    lock_acquisitions = []

    def worker(thread_id):
        try:
            cache.acquire_lock()
            lock_acquisitions.append(thread_id)
            time.sleep(2)  # Hold lock for 2 seconds
            cache.release_lock()
        except Exception as e:
            pytest.fail(f"Thread {thread_id} failed: {e}")

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    time.sleep(0.2)
    t2.start()

    t1.join()
    t2.join()

    assert not t1.is_alive() and not t2.is_alive()

    assert len(lock_acquisitions) == 2, (
        f"Expected 2 lock acquisitions, got {len(lock_acquisitions)}"
    )
    assert 1 in lock_acquisitions, "Thread 1 did not acquire lock"
    assert 2 in lock_acquisitions, "Thread 2 did not acquire lock"

    cache.cache.close()


def test_two_processes():
    process_1 = Process(target=_process_worker, args=("cognee_process_1",))
    process_2 = Process(target=_process_worker, args=("cognee_process_2",))

    process_1.start()
    process_2.start()

    process_1.join()
    process_2.join()

    assert process_1.exitcode == 0, f"Process 1 failed with exit code {process_1.exitcode}"
    assert process_2.exitcode == 0, f"Process 2 failed with exit code {process_2.exitcode}"


def test_timeout_while_lock_held():
    """Test that acquisition timeout works when lock is held by another process."""
    lock_key = "test_timeout_held"

    cache1 = FSCacheAdapter(timeout=10, blocking_timeout=None, lock_key=lock_key)
    cache1.acquire_lock()

    try:
        cache2 = FSCacheAdapter(timeout=2, blocking_timeout=None, lock_key=lock_key)

        start_time = time.time()
        with pytest.raises(TimeoutError) as exc_info:
            cache2.acquire_lock()

        elapsed = time.time() - start_time

        assert "Failed to acquire lock within 2 seconds" in str(exc_info.value)
        assert 1.9 < elapsed < 2.5, f"Expected timeout ~2s, got {elapsed:.2f}s"

        cache2.cache.close()
    finally:
        cache1.release_lock()
        cache1.cache.close()


@pytest.mark.asyncio
async def test_closing_connection():
    cache = FSCacheAdapter(timeout=1, blocking_timeout=10, lock_key="test_closing")
    await cache.close()
