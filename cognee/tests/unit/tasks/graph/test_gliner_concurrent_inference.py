"""Concurrent GLiNER inference: identical output, bounded by CPU and memory.

The equivalence tests run the ``gliner2`` runtime's real long-text code (split,
batching, merge) with only the model's forward pass replaced, so they compare
the concurrent path against the exact sequential call cognee made before.
They need ``gliner2`` installed and are skipped otherwise.
"""

import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cognee.tasks.graph.gliner_demo import extractor as extractor_module

# Captured at import, before the directory's autouse fixture pins one thread.
real_inference_threads = extractor_module.inference_threads
GIB = 1024**3


def names_text(seed: int, words: int) -> str:
    vocabulary = ["Pierre", "Natasha", "Moscow", "walked", "with", "the", "Prince", "and", "army"]
    return " ".join(vocabulary[(seed * 7 + i * 3) % len(vocabulary)] for i in range(words))


def window_model():
    gliner_runtime = pytest.importorskip("gliner2.inference.runtime")

    class WindowModel(gliner_runtime.ExtractorRuntimeMixin):
        """The runtime's real long-text path; only the model forward pass is faked."""

        architecture = "span"

        def __init__(self):
            self.batches: list[list[str]] = []
            self.threads: set[str] = set()
            self._lock = threading.Lock()

        def batch_extract(self, texts, schemas, **_options):
            with self._lock:
                self.batches.append(list(texts))
                self.threads.add(threading.current_thread().name)
            time.sleep(0.02)  # long enough for concurrent batches to overlap
            return [self.forward(text) for text in texts]

        @staticmethod
        def forward(text):
            entities = [
                {
                    "text": match.group(),
                    "confidence": 0.5 + (sum(map(ord, match.group())) % 50) / 100,
                    "start": match.start(),
                    "end": match.end(),
                }
                for match in re.finditer(r"\b[A-Z][a-z]+\b", text)
            ]
            return {"entities": {"name": entities}}

    return WindowModel()


OPTIONS = {"threshold": 0.5, "batch_size": 4, "window_words": 60, "window_overlap_words": 12}
TEXTS = [names_text(seed, words) for seed, words in [(1, 400), (2, 35), (3, 900), (4, 0)]]


def sequential_reference(model):
    """Exactly the call cognee made before: the runtime's own long-text path."""
    return model.batch_extract_long(
        TEXTS,
        object(),
        batch_size=OPTIONS["batch_size"],
        threshold=OPTIONS["threshold"],
        include_confidence=True,
        include_spans=True,
        chunk_size=OPTIONS["window_words"],
        chunk_overlap=OPTIONS["window_overlap_words"],
        overlap_policy=extractor_module.OVERLAP_POLICY,
    )


def test_concurrent_path_matches_the_runtimes_sequential_long_text_path():
    reference = sequential_reference(window_model())
    model = window_model()
    with ThreadPoolExecutor(4) as pool:
        concurrent = extractor_module._extract_long_concurrently(
            model, pool, TEXTS, object(), **OPTIONS
        )
    assert concurrent == reference
    assert len(model.threads) > 1, "batches did not run on more than one thread"


def test_every_batch_keeps_its_single_threaded_members():
    """Same windows, grouped the same way: the property that keeps scores bit-identical."""
    model = window_model()  # skips first when gliner2 is not installed
    from gliner2.inference.chunking import split_text_into_chunks

    windows = [
        window.text
        for text in TEXTS
        for window in split_text_into_chunks(
            text, chunk_size=OPTIONS["window_words"], chunk_overlap=OPTIONS["window_overlap_words"]
        )
    ]
    size = OPTIONS["batch_size"]
    expected = [windows[start : start + size] for start in range(0, len(windows), size)]
    with ThreadPoolExecutor(3) as pool:
        extractor_module._extract_long_concurrently(model, pool, TEXTS, object(), **OPTIONS)
    assert sorted(model.batches) == sorted(expected)


@pytest.fixture
def machine(monkeypatch):
    """Stand-ins for torch and psutil: a machine with a given size."""

    def configure(torch_threads: int, available_gib: float):
        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: torch_threads)
        )
        memory = SimpleNamespace(available=int(available_gib * GIB))
        monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: memory))

    return configure


@pytest.mark.parametrize(
    ("torch_threads", "available_gib", "expected"),
    [
        (10, 64, 5),  # CPU-bound: half of torch's thread count
        (14, 64, 7),
        (10, 9, 3),  # memory-bound: 1 + (9 - 4 reserve) // 2
        (10, 5, 1),  # barely above the reserve: stay single-threaded
        (10, 1, 1),  # below the reserve: never zero
        (1, 64, 1),  # one core
    ],
)
def test_auto_sizing_takes_the_tighter_of_cpu_and_memory(
    machine, torch_threads, available_gib, expected
):
    machine(torch_threads, available_gib)
    assert extractor_module.auto_inference_threads() == expected


def test_larger_batches_count_for_more_memory(machine):
    machine(10, 12)
    assert extractor_module.auto_inference_threads(batch_size=16) == 5
    assert extractor_module.auto_inference_threads(batch_size=32) == 3


def test_configured_thread_count_wins_over_auto_sizing(monkeypatch):
    monkeypatch.setattr(
        extractor_module,
        "get_cognify_config",
        lambda: SimpleNamespace(gliner_inference_threads=3),
    )
    monkeypatch.setattr(extractor_module, "auto_inference_threads", lambda *_: pytest.fail())
    assert real_inference_threads() == 3


def test_zero_means_auto(monkeypatch):
    monkeypatch.setattr(
        extractor_module, "get_cognify_config", lambda: SimpleNamespace(gliner_inference_threads=0)
    )
    monkeypatch.setattr(extractor_module, "auto_inference_threads", lambda *_: 4)
    assert real_inference_threads() == 4


def test_negative_thread_count_is_rejected(monkeypatch):
    monkeypatch.setattr(
        extractor_module, "get_cognify_config", lambda: SimpleNamespace(gliner_inference_threads=-1)
    )
    with pytest.raises(ValueError, match="GLINER_INFERENCE_THREADS"):
        real_inference_threads()


def test_one_thread_keeps_the_runtimes_long_text_call():
    """The single-threaded path is the unchanged public call, under the lock."""
    extractor_module.reset_inference_pool()
    assert extractor_module._inference_pool(16) is None


def test_several_threads_share_one_pool(monkeypatch):
    monkeypatch.setattr(extractor_module, "inference_threads", lambda batch_size=16: 3)
    extractor_module.reset_inference_pool()
    first = extractor_module._inference_pool(16)
    assert first is not None and first._max_workers == 3
    assert extractor_module._inference_pool(32) is first, "one pool per process"
