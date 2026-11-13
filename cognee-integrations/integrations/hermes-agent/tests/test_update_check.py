"""The PyPI update check: cached, rate-limited, fail-silent.

Run standalone with ``python3 tests/test_update_check.py``.
"""

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import update_check  # noqa: E402


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener_returning(version, calls=None):
    def opener(request, timeout=None):
        if calls is not None:
            calls.append(request.full_url)
        return _Response(json.dumps({"info": {"version": version}}).encode("utf-8"))

    return opener


class TestLatestPublishedVersion(unittest.TestCase):
    def _cache(self, tmp_name="update-check.json"):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name) / tmp_name

    def test_fetches_and_caches(self):
        cache = self._cache()
        calls = []
        latest = update_check.latest_published_version(
            cache_path=cache, opener=_opener_returning("2.0.0", calls), now=1000.0
        )
        self.assertEqual(latest, "2.0.0")
        self.assertEqual(len(calls), 1)
        stored = json.loads(cache.read_text())
        self.assertEqual(stored["latest"], "2.0.0")
        self.assertEqual(stored["checked_at"], 1000.0)

    def test_a_fresh_cache_skips_the_network(self):
        cache = self._cache()
        calls = []
        opener = _opener_returning("2.0.0", calls)
        update_check.latest_published_version(cache_path=cache, opener=opener, now=1000.0)
        latest = update_check.latest_published_version(
            cache_path=cache, opener=opener, now=1000.0 + 100, interval=3600
        )
        self.assertEqual(latest, "2.0.0")
        self.assertEqual(len(calls), 1)

    def test_an_expired_cache_rechecks(self):
        cache = self._cache()
        calls = []
        opener = _opener_returning("2.0.0", calls)
        update_check.latest_published_version(cache_path=cache, opener=opener, now=1000.0)
        update_check.latest_published_version(
            cache_path=cache, opener=opener, now=1000.0 + 7200, interval=3600
        )
        self.assertEqual(len(calls), 2)

    def test_force_skips_the_rate_limit(self):
        cache = self._cache()
        calls = []
        opener = _opener_returning("2.0.0", calls)
        update_check.latest_published_version(cache_path=cache, opener=opener, now=1000.0)
        update_check.latest_published_version(
            cache_path=cache, opener=opener, now=1000.0 + 1, force=True
        )
        self.assertEqual(len(calls), 2)

    def test_a_network_failure_returns_the_cached_answer(self):
        cache = self._cache()
        update_check.latest_published_version(
            cache_path=cache, opener=_opener_returning("2.0.0"), now=1000.0
        )

        def broken(request, timeout=None):
            raise OSError("offline")

        latest = update_check.latest_published_version(
            cache_path=cache, opener=broken, now=1000.0 + 7200, interval=3600
        )
        self.assertEqual(latest, "2.0.0")

    def test_a_network_failure_with_no_cache_returns_empty(self):
        def broken(request, timeout=None):
            raise OSError("offline")

        latest = update_check.latest_published_version(cache_path=self._cache(), opener=broken)
        self.assertEqual(latest, "")


class TestIsNewer(unittest.TestCase):
    def test_ordering(self):
        self.assertTrue(update_check.is_newer("1.2.0", "1.1.0"))
        self.assertTrue(update_check.is_newer("1.10.0", "1.9.9"))
        self.assertFalse(update_check.is_newer("1.1.0", "1.1.0"))
        self.assertFalse(update_check.is_newer("1.0.9", "1.1.0"))

    def test_blank_sides_never_nag(self):
        self.assertFalse(update_check.is_newer("", "1.0.0"))
        self.assertFalse(update_check.is_newer("1.0.0", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
