"""The overflow scan: the server log is the only trace of a lossy embedding.

cognee's OllamaEmbeddingEngine mean-pools a text that overflows the embedding
model's context and reports success — the write looks fine while the vector
index quietly degrades. The only evidence is an "Ollama embedding error" line in
the spawned server's log, so ``HttpBackend`` tails that log between calls and
``overflow_hint()`` turns a fresh error into an actionable message the provider
puts in the tool envelope (envelope shapes: ``test_tool_contract.py``).

Runs standalone with ``python3 tests/test_overflow_hint.py``; needs no cognee.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import http_backend as hb  # noqa: E402
from cognee_integration_hermes.http_backend import HttpBackend  # noqa: E402

_LOCAL_URL = "http://127.0.0.1:8011"
_OVERFLOW_LINE = b"ERROR Ollama embedding error: input length exceeds maximum context length\n"


class TestOverflowHint(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "server.log"

    def _armed(self, *, url=_LOCAL_URL, prior=b""):
        """A backend whose scan was armed (as ``connect()`` does) at the current
        end of a log holding *prior* bytes."""
        self.log.write_bytes(prior)
        backend = HttpBackend(server_log_path=str(self.log))
        backend.url = url
        backend._init_overflow_scan()
        return backend

    def _append(self, data):
        with open(self.log, "ab") as log:
            log.write(data)

    def test_a_fresh_overflow_line_fires_an_actionable_hint(self):
        backend = self._armed()
        self._append(_OVERFLOW_LINE)
        hint = backend.overflow_hint()
        self.assertIsNotNone(hint)
        # The hint must name the levers that fix it, not just describe the problem.
        self.assertIn("EMBEDDING_MAX_COMPLETION_TOKENS", hint)
        self.assertIn("HUGGINGFACE_TOKENIZER", hint)

    def test_unrelated_log_lines_do_not_fire(self):
        backend = self._armed()
        self._append(b"INFO uvicorn running\nINFO cognify pipeline completed\n")
        self.assertIsNone(backend.overflow_hint())

    def test_errors_from_before_the_connect_never_fire(self):
        # The offset snapshots the log's end when the scan is armed, so damage
        # left behind by earlier sessions cannot fire a hint in this one.
        backend = self._armed(prior=_OVERFLOW_LINE)
        self.assertIsNone(backend.overflow_hint())

    def test_the_snapshot_sits_at_the_current_end_of_the_log(self):
        backend = self._armed(prior=b"12345")
        self.assertEqual(backend._log_offset, 5)

    def test_one_error_batch_fires_exactly_once(self):
        backend = self._armed()
        self._append(b"Text too long for embedding model: 4000 tokens\n")
        self.assertIsNotNone(backend.overflow_hint())
        self.assertIsNone(backend.overflow_hint())
        self._append(_OVERFLOW_LINE)
        self.assertIsNotNone(backend.overflow_hint())

    def test_a_clean_scan_still_advances_the_offset(self):
        backend = self._armed()
        self._append(b"INFO noise\n")
        self.assertIsNone(backend.overflow_hint())
        self._append(_OVERFLOW_LINE)
        self.assertIsNotNone(backend.overflow_hint())

    def test_matching_is_case_insensitive(self):
        backend = self._armed()
        self._append(b"OLLAMA EMBEDDING ERROR: boom\n")
        self.assertIsNotNone(backend.overflow_hint())

    def test_a_missing_log_is_a_quiet_none(self):
        backend = HttpBackend(server_log_path=str(self.log.parent / "nope" / "server.log"))
        backend.url = _LOCAL_URL
        backend._init_overflow_scan()
        self.assertIsNone(backend.overflow_hint())

    def test_a_remote_url_never_scans(self):
        # A remote server's log is not on this filesystem; scanning a local file
        # would attribute someone else's errors to it.
        backend = self._armed(url="https://cloud.example")
        self._append(_OVERFLOW_LINE)
        self.assertIsNone(backend.overflow_hint())

    def test_an_unarmed_backend_returns_none(self):
        backend = HttpBackend(server_log_path=str(self.log))
        self.assertIsNone(backend.overflow_hint())

    def test_a_truncated_log_rescans_from_the_top(self):
        backend = self._armed(prior=b"x" * 100)
        self.log.write_bytes(_OVERFLOW_LINE)  # rotation: smaller than the offset
        self.assertIsNotNone(backend.overflow_hint())

    def test_a_giant_append_is_capped_but_the_tail_is_still_scanned(self):
        backend = self._armed()
        self._append(b"a" * (hb._OVERFLOW_SCAN_CAP + 4096))
        self._append(_OVERFLOW_LINE)
        self.assertIsNotNone(backend.overflow_hint())


if __name__ == "__main__":
    unittest.main(verbosity=2)
