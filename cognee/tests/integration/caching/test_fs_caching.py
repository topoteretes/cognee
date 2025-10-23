import pytest
import threading
import time

from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter


def test_can_be_instantiated():
    FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="lock_key_name")


def test_can_acquire_lock():
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="lock_key_name")
    cache.acquire_lock()


def test_can_release_lock():
    cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="lock_key_name")
    cache.acquire_lock()
    cache.release_lock()


def test_releasing_non_acquired_lock():
    try:
        cache = FSCacheAdapter(timeout=5, blocking_timeout=10, lock_key="lock_key_name")
        cache.acquire_lock()
        cache.release_lock()
    except Exception as e:
        pytest.fail(f"Failed: {e}")


@pytest.mark.timeout(10)
def test_two_threads():
    cache = FSCacheAdapter(timeout=1, blocking_timeout=10, lock_key="lock_key_name")
    lock_acquisitions = []

    def worker(thread_id):
        try:
            cache.acquire_lock()
            lock_acquisitions.append(thread_id)
            time.sleep(2)
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

    # Verify both threads completed
    assert not t1.is_alive() and not t2.is_alive()

    # Verify both threads successfully acquired the lock
    assert len(lock_acquisitions) == 2, (
        f"Expected 2 lock acquisitions, got {len(lock_acquisitions)}"
    )
    assert 1 in lock_acquisitions, "Thread 1 did not acquire lock"
    assert 2 in lock_acquisitions, "Thread 2 did not acquire lock"


def test_closing_connection():
    cache = FSCacheAdapter(timeout=1, blocking_timeout=10, lock_key="lock_key_name")
    cache.close()
