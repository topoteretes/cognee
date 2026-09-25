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

        def create_schema(self):
            """What a worker process uses to rebuild the schema from plain data."""
            return SimpleNamespace(entities=lambda _spec: None, relations=lambda _spec: None)

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
            model, TEXTS, object(), thread_pool=pool, **OPTIONS
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
        extractor_module._extract_long_concurrently(
            model, TEXTS, object(), thread_pool=pool, **OPTIONS
        )
    assert sorted(model.batches) == sorted(expected)


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    """A fake /sys/fs/cgroup: write the files a container would have."""
    monkeypatch.setattr(extractor_module, "CGROUP_ROOT", tmp_path)

    def write(name: str, value: str):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")

    return write


@pytest.fixture
def machine(monkeypatch, cgroup):
    """Stand-ins for torch and psutil: a machine with a given size, no container limits."""

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
    monkeypatch.setattr(extractor_module, "inference_threads", lambda *_args, **_kwargs: 3)
    extractor_module.reset_inference_pool()
    first = extractor_module._inference_pool(16)
    assert first is not None and first._max_workers == 3
    assert extractor_module._inference_pool(32) is first, "one pool per process"


def test_no_cgroup_files_means_no_container_limits(cgroup):
    """macOS, Windows, and Linux outside a container: nothing is read as a limit."""
    assert extractor_module.container_cpu_limit() is None
    assert extractor_module.container_memory_free() is None


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"cpu.max": "200000 100000"}, 2.0),  # docker run --cpus=2, cgroup v2
        ({"cpu.max": "150000 100000"}, 1.5),
        ({"cpu.max": "max 100000"}, None),  # v2, no quota
        ({"cpu/cpu.cfs_quota_us": "300000", "cpu/cpu.cfs_period_us": "100000"}, 3.0),  # v1
        ({"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, None),  # v1, none
    ],
)
def test_container_cpu_quota(cgroup, files, expected):
    for name, value in files.items():
        cgroup(name, value)
    assert extractor_module.container_cpu_limit() == expected


@pytest.mark.parametrize(
    ("files", "expected_gib"),
    [
        ({"memory.max": str(4 * GIB), "memory.current": str(1 * GIB)}, 3),  # --memory=4g, v2
        ({"memory.max": "max", "memory.current": str(1 * GIB)}, None),  # v2, no limit
        (
            {
                "memory/memory.limit_in_bytes": str(6 * GIB),
                "memory/memory.usage_in_bytes": str(2 * GIB),
            },
            4,
        ),  # v1
        (
            {
                "memory/memory.limit_in_bytes": "9223372036854771712",
                "memory/memory.usage_in_bytes": str(GIB),
            },
            None,
        ),  # v1 reports "no limit" as ~2**63
        ({"memory.max": str(2 * GIB), "memory.current": str(3 * GIB)}, 0),  # over the limit
    ],
)
def test_container_memory_left(cgroup, files, expected_gib):
    for name, value in files.items():
        cgroup(name, value)
    free = extractor_module.container_memory_free()
    assert free == (None if expected_gib is None else expected_gib * GIB)


def test_a_small_container_on_a_big_host_is_sized_to_the_container(machine, cgroup):
    """What Docker actually exposes, measured: the host's CPUs and free memory
    stay visible inside `docker run --cpus=2 --memory=4g`; only cgroups hold the
    limits. Sizing from the host would run 7 batches in a 4 GB container."""
    machine(14, 16.6)
    assert extractor_module.auto_inference_threads() == 7
    cgroup("cpu.max", "200000 100000")
    cgroup("memory.max", str(4 * GIB))
    cgroup("memory.current", str(int(0.2 * GIB)))
    assert extractor_module.auto_inference_threads() == 1


def test_a_container_quota_and_limit_both_bound_the_pool(machine, cgroup):
    machine(32, 100)
    cgroup("cpu.max", "800000 100000")  # 8 CPUs: CPU bound 4
    cgroup("memory.max", str(12 * GIB))
    cgroup("memory.current", str(1 * GIB))  # 11 GiB left: memory bound 1 + (11 - 4) // 2 = 4
    assert extractor_module.auto_inference_threads() == 4
    cgroup("memory.current", str(6 * GIB))  # 6 GiB left: memory bound 1 + 2 // 2 = 2
    assert extractor_module.auto_inference_threads() == 2


def test_a_failing_batch_raises_and_the_pool_keeps_working():
    """An error in one batch reaches the caller, and the shared pool stays usable."""
    model = window_model()
    real_batch_extract = model.batch_extract

    def failing(texts, schemas, **options):
        if any("Moscow" in text for text in texts):
            raise RuntimeError("model failed on this batch")
        return real_batch_extract(texts, schemas, **options)

    model.batch_extract = failing
    with ThreadPoolExecutor(3) as pool:
        with pytest.raises(RuntimeError, match="model failed on this batch"):
            extractor_module._extract_long_concurrently(
                model, TEXTS, object(), thread_pool=pool, **OPTIONS
            )
        model.batch_extract = real_batch_extract
        recovered = extractor_module._extract_long_concurrently(
            model, TEXTS, object(), thread_pool=pool, **OPTIONS
        )
    assert recovered == sequential_reference(window_model())


def test_concurrent_callers_share_the_pool_without_mixing_results():
    """Two documents extracted at once through one pool each get their own result."""
    other_texts = [names_text(seed, 300) for seed in (7, 8, 9)]
    expected_a = sequential_reference(window_model())
    expected_b = window_model().batch_extract_long(
        other_texts,
        object(),
        batch_size=OPTIONS["batch_size"],
        threshold=OPTIONS["threshold"],
        include_confidence=True,
        include_spans=True,
        chunk_size=OPTIONS["window_words"],
        chunk_overlap=OPTIONS["window_overlap_words"],
        overlap_policy=extractor_module.OVERLAP_POLICY,
    )
    model = window_model()
    with ThreadPoolExecutor(4) as pool, ThreadPoolExecutor(2) as callers:
        future_a = callers.submit(
            extractor_module._extract_long_concurrently,
            model,
            TEXTS,
            object(),
            thread_pool=pool,
            **OPTIONS,
        )
        future_b = callers.submit(
            extractor_module._extract_long_concurrently,
            model,
            other_texts,
            object(),
            thread_pool=pool,
            **OPTIONS,
        )
        assert future_a.result() == expected_a
        assert future_b.result() == expected_b


def test_repeated_extraction_does_not_grow_the_thread_count(monkeypatch):
    """The process-wide pool is reused: many calls never exceed its size in threads."""
    monkeypatch.setattr(extractor_module, "inference_threads", lambda *_args, **_kwargs: 3)
    extractor_module.reset_inference_pool()
    pool = extractor_module._inference_pool(OPTIONS["batch_size"])
    model = window_model()
    for _ in range(10):
        extractor_module._extract_long_concurrently(
            model, TEXTS, object(), thread_pool=pool, **OPTIONS
        )
    gliner_threads = [t for t in threading.enumerate() if t.name.startswith("gliner")]
    assert 0 < len(gliner_threads) <= 3
    extractor_module.reset_inference_pool()
    time.sleep(0.1)
    assert not [t for t in threading.enumerate() if t.name.startswith("gliner")]


def test_worker_processes_take_every_nth_batch_and_the_result_is_unchanged(monkeypatch):
    """processes=3: this process keeps every 3rd batch, workers run the rest, and the
    merged result equals the runtime's sequential path. A thread pool stands in for
    the worker processes, running the real worker entry point on a fake model."""
    local, remote = window_model(), window_model()  # skips first without gliner2
    from gliner2.inference.chunking import split_text_into_chunks

    from cognee.tasks.graph.gliner_demo.schema import GlinerSchema

    monkeypatch.setattr(extractor_module, "_worker", {"extractor": remote, "pool": None})
    with ThreadPoolExecutor(2) as stand_in_for_processes:
        result = extractor_module._extract_long_concurrently(
            local,
            TEXTS,
            object(),
            schema=GlinerSchema({"name": ""}, {}, source="caller"),
            process_pool=stand_in_for_processes,
            processes=3,
            **OPTIONS,
        )
    assert result == sequential_reference(window_model())

    windows = [
        window.text
        for text in TEXTS
        for window in split_text_into_chunks(
            text, chunk_size=OPTIONS["window_words"], chunk_overlap=OPTIONS["window_overlap_words"]
        )
    ]
    size = OPTIONS["batch_size"]
    batches = [windows[start : start + size] for start in range(0, len(windows), size)]
    assert local.batches == batches[0::3], "this process keeps every third batch, in order"
    assert sorted(remote.batches) == sorted(batches[1::3] + batches[2::3])


@pytest.mark.parametrize(
    ("torch_threads", "available_gib", "processes", "expected"),
    [
        (10, 64, 1, 5),  # one process: the machine's 5 concurrent batches
        (10, 64, 2, 2),  # split across 2 processes: 5 // 2
        (10, 64, 3, 1),
        (10, 12, 2, 1),  # the 2nd model copy costs 3 GiB: 1 + (12 - 3 - 4) // 2 = 3, // 2 = 1
        (14, 64, 2, 3),
    ],
)
def test_auto_sizing_divides_the_machine_across_processes(
    machine, torch_threads, available_gib, processes, expected
):
    machine(torch_threads, available_gib)
    assert extractor_module.auto_inference_threads(processes=processes) == expected


def test_process_count_defaults_to_one_and_a_request_wins(monkeypatch):
    monkeypatch.setattr(
        extractor_module,
        "get_cognify_config",
        lambda: SimpleNamespace(gliner_inference_processes=1, gliner_inference_threads=0),
    )
    assert extractor_module.inference_processes() == 1
    assert extractor_module.inference_processes(3) == 3
    with pytest.raises(ValueError, match="processes must be >= 1"):
        extractor_module.inference_processes(0)


def test_requested_threads_win_over_the_configured_value(monkeypatch):
    monkeypatch.setattr(
        extractor_module,
        "get_cognify_config",
        lambda: SimpleNamespace(gliner_inference_threads=5),
    )
    assert real_inference_threads(requested=2) == 2
    assert real_inference_threads() == 5


def test_concurrency_is_resolved_once_per_call_shape(monkeypatch):
    """Auto-sizing reads free memory, which the pools then use: the first answer sticks."""
    calls = []

    def counting_threads(batch_size, processes, requested):
        calls.append((batch_size, processes, requested))
        return 2

    monkeypatch.setattr(extractor_module, "inference_threads", counting_threads)
    monkeypatch.setattr(
        extractor_module, "inference_processes", lambda requested=None: requested or 1
    )
    extractor_module.reset_inference_pool()
    assert extractor_module._concurrency(16) == (1, 2)
    assert extractor_module._concurrency(16) == (1, 2)
    assert extractor_module._concurrency(16, 2, 3) == (2, 2)
    assert calls == [(16, 1, None), (16, 2, 3)]
    extractor_module.reset_inference_pool()


def test_a_broken_worker_pool_is_dropped_so_the_next_call_starts_fresh(monkeypatch):
    """A worker killed for memory breaks its pool for good; it must not poison later calls."""
    from concurrent.futures.process import BrokenProcessPool

    from cognee.tasks.graph.gliner_demo.schema import GlinerSchema

    extractor_module.reset_inference_pool()
    monkeypatch.setattr(extractor_module, "_concurrency", lambda *_args: (2, 1))
    monkeypatch.setattr(extractor_module, "build_gliner_schema", lambda *_args: object())

    def broken(*_args, **_kwargs):
        raise BrokenProcessPool("a worker died")

    monkeypatch.setattr(extractor_module, "_extract_long_concurrently", broken)
    sentinel = SimpleNamespace(shutdown=lambda **_kwargs: None)
    extractor_module._process_pools[("model", 1, 1)] = sentinel

    with pytest.raises(BrokenProcessPool):
        extractor_module.extract_batch(
            object(), ["text"], GlinerSchema({"name": ""}, {}, source="caller"), model_name="model"
        )
    assert ("model", 1, 1) not in extractor_module._process_pools
