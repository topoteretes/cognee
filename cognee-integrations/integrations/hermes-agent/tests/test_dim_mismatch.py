"""Tests for the embedding-dimension probe in the SDK transport.

The probe lives with ``SdkBackend`` because it introspects the in-process vector
engine — see ``backend.py``. The provider only asks for a hint when a recall comes
back empty; the transport decides whether it can answer.

``dimension_mismatch_hint`` returns a one-line actionable diagnostic naming both
dims and the active model on a confirmed mismatch, and None in every other case
(matching dims, no data, or any error) so recall keeps its normal empty-result
behavior. A fake vector engine is injected so cognee is not required.

Runs under pytest or standalone (``python3 tests/test_dim_mismatch.py``).
"""

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import backend as backend_mod  # noqa: E402


class _FakeEmbed:
    def __init__(self, size, model="openai/text-embedding-3-large", provider="openai"):
        self._size = size
        self.model = model
        self.provider = provider

    def get_vector_size(self):
        return self._size


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def limit(self, _n):
        return self

    async def to_list(self):
        return self._rows


class _FakeCollection:
    def __init__(self, rows):
        self._rows = rows

    def query(self):
        return _FakeQuery(self._rows)


class _FakeConnection:
    def __init__(self, names):
        self._names = list(names)

    async def table_names(self):
        return self._names


class _FakeLanceEngine:
    def __init__(self, query_size, stored_vector, present=("Entity_name",)):
        self.embedding_engine = _FakeEmbed(query_size)
        self._stored_vector = stored_vector
        self._present = tuple(present)

    async def get_connection(self):
        return _FakeConnection(self._present)

    async def get_collection(self, _name):
        rows = [{"vector": self._stored_vector}] if self._stored_vector is not None else []
        return _FakeCollection(rows)


class _FakeErrorEngine:
    def __init__(self, query_size):
        self.embedding_engine = _FakeEmbed(query_size)

    async def get_connection(self):
        return _FakeConnection(("Entity_name",))

    async def get_collection(self, _name):
        raise RuntimeError("store locked")


def _hint(engine):
    return asyncio.run(backend_mod.dimension_mismatch_hint(engine))


class TestDimensionMismatchHint(unittest.TestCase):
    def test_mismatch_names_both_dims_and_model(self):
        msg = _hint(_FakeLanceEngine(query_size=3072, stored_vector=[0.0] * 1536))
        self.assertIsNotNone(msg)
        self.assertIn("1536", msg)
        self.assertIn("3072", msg)
        self.assertIn("text-embedding-3-large", msg)

    def test_matching_dims_returns_none(self):
        self.assertIsNone(_hint(_FakeLanceEngine(query_size=1536, stored_vector=[0.0] * 1536)))

    def test_enumerates_nonstandard_collection(self):
        # Enumeration (not a fixed name list) means a custom-pipeline collection is
        # still sampled, so the diagnostic fires for non-standard stores too.
        engine = _FakeLanceEngine(
            query_size=3072, stored_vector=[0.0] * 1536, present=("CustomThing_body",)
        )
        self.assertIsNotNone(_hint(engine))

    def test_no_collections_returns_none(self):
        engine = _FakeLanceEngine(query_size=3072, stored_vector=[0.0] * 1536, present=())
        self.assertIsNone(_hint(engine))

    def test_unreadable_store_returns_none(self):
        # A read error must not surface a hint (no false alarm on a healthy store).
        self.assertIsNone(_hint(_FakeErrorEngine(query_size=3072)))

    def test_broken_engine_returns_none(self):
        class _Broken:
            @property
            def embedding_engine(self):
                raise RuntimeError("boom")

        self.assertIsNone(_hint(_Broken()))


class TestBackendGate(unittest.TestCase):
    def test_served_mode_skips_the_check(self):
        backend = backend_mod.SdkBackend()
        backend.served = True
        self.assertIsNone(backend.empty_recall_hint())

    def test_in_process_mode_runs_the_check(self):
        backend = backend_mod.SdkBackend()
        backend.served = False

        async def _fake_hint():
            return "DIM MISMATCH"

        original = backend_mod.dimension_mismatch_hint
        backend_mod.dimension_mismatch_hint = _fake_hint
        try:
            self.assertEqual(backend.empty_recall_hint(), "DIM MISMATCH")
        finally:
            backend_mod.dimension_mismatch_hint = original
            backend.close()


if __name__ == "__main__":
    unittest.main()
