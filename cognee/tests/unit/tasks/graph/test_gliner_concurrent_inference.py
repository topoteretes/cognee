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
def cgroup(tmp_path, monkeypatch):
    """A fake /sys/fs/cgroup and /proc/self/cgroup.

    ``cgroup(name, value)`` writes one file under the cgroup mount;
    ``cgroup.process(*lines)`` sets the process's /proc/self/cgroup lines. The
    default is Docker's private namespace: the process at the root of a v2
    hierarchy, ``0::/``.
    """
    root = tmp_path / "sys_fs_cgroup"
    proc = tmp_path / "proc_self_cgroup"
    monkeypatch.setattr(extractor_module, "CGROUP_ROOT", root)
    monkeypatch.setattr(extractor_module, "PROC_SELF_CGROUP", proc)
    proc.write_text("0::/\n")

    def write(name: str, value: str):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")

    write.process = lambda *lines: proc.write_text("\n".join(lines) + "\n")
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


def test_sizing_assumes_the_default_batch_size(machine):
    """The pool is sized once, by its first user, which is the schema probe:
    it has no batch size of its own, so sizing is pinned to DEFAULT_BATCH_SIZE
    rather than taken from whichever call arrives first. A larger
    gliner_batch_size needs GLINER_INFERENCE_THREADS set by hand."""
    machine(10, 12)
    assert extractor_module.auto_inference_threads() == 5
    assert extractor_module.DEFAULT_BATCH_SIZE == 16


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


def test_negative_thread_count_is_rejected():
    from cognee.modules.cognify.config import CognifyConfig

    # pydantic's ValidationError is a ValueError; it is raised when the config loads.
    with pytest.raises(ValueError, match="gliner_inference_threads"):
        CognifyConfig(gliner_inference_threads=-1)


class OverlapCountingModel:
    """A fake model that records the most calls ever running on it at once."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def _call(self):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)  # long enough for unguarded calls to overlap
        with self._lock:
            self.active -= 1

    def batch_extract_long(self, texts, *_args, **_options):
        self._call()
        return [{} for _ in texts]

    def extract(self, *_args, **_options):
        self._call()
        return {}


def test_one_thread_keeps_the_runtimes_long_text_call():
    """The single-threaded path is the unchanged public call, batches run in turn."""
    extractor_module.reset_inference_pool()
    assert extractor_module._inference_pool() is None


def test_one_thread_runs_one_model_call_at_a_time(monkeypatch):
    """With no pool, concurrent pipelines take turns, which bounds their memory."""
    monkeypatch.setattr(extractor_module, "build_gliner_schema", lambda *_: object())
    model, schema = OverlapCountingModel(), SimpleNamespace(is_empty=False)
    with ThreadPoolExecutor(4) as callers:
        for _ in range(2):
            callers.submit(extractor_module.extract_batch, model, ["text"], schema)
            callers.submit(extractor_module.extract_once, model, "text", schema)
    assert model.peak == 1


def test_several_threads_share_one_pool(monkeypatch):
    monkeypatch.setattr(extractor_module, "inference_threads", lambda: 3)
    extractor_module.reset_inference_pool()
    first = extractor_module._inference_pool()
    assert first is not None and first._max_workers == 3
    assert extractor_module._inference_pool() is first, "one pool per process"


def test_no_cgroup_files_means_no_container_limits(cgroup):
    """Linux outside a container: the files exist but hold no limit, or are absent."""
    assert extractor_module.container_cpu_limit() is None
    assert extractor_module.container_memory_free() is None


def test_no_proc_self_cgroup_means_no_container_limits(cgroup):
    """macOS and Windows have no /proc/self/cgroup, so there is nothing to read."""
    extractor_module.PROC_SELF_CGROUP.unlink()
    cgroup("cpu.max", "200000 100000")  # never consulted without a cgroup path
    assert extractor_module._cgroup_dirs("cpu") == []
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
    if any(name.startswith("cpu/") for name in files):
        cgroup.process("3:cpu:/")
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
    if any(name.startswith("memory/") for name in files):
        cgroup.process("5:memory:/")
    for name, value in files.items():
        cgroup(name, value)
    free = extractor_module.container_memory_free()
    assert free == (None if expected_gib is None else expected_gib * GIB)


def test_process_cgroup_path_is_resolved_from_proc_self_cgroup(cgroup):
    """Docker --cgroupns=host, measured: the process is at /docker/<id>, not at
    the root, and the limit files are in that directory. The root has none."""
    cgroup.process("0::/docker/80e6330e3f72")
    cgroup("docker/80e6330e3f72/cpu.max", "200000 100000")
    cgroup("docker/80e6330e3f72/memory.max", str(4 * GIB))
    cgroup("docker/80e6330e3f72/memory.current", str(1 * GIB))
    assert extractor_module._cgroup_dirs("cpu") == [
        extractor_module.CGROUP_ROOT / "docker/80e6330e3f72",
        extractor_module.CGROUP_ROOT / "docker",
        extractor_module.CGROUP_ROOT,
    ]
    assert extractor_module.container_cpu_limit() == 2.0
    assert extractor_module.container_memory_free() == 3 * GIB


def test_a_limit_on_an_ancestor_cgroup_binds_the_process(cgroup):
    """A Kubernetes pod on a cgroup v2 node: the container's own cgroup has no
    memory limit, the pod slice above it does, and a looser CPU quota sits on
    the container than on the pod. The tightest level wins for each resource."""
    pod = "kubepods.slice/kubepods-pod1.slice"
    container = f"{pod}/cri-containerd-abc.scope"
    cgroup.process(f"0::/{container}")
    cgroup(f"{container}/cpu.max", "400000 100000")
    cgroup(f"{container}/memory.max", "max")
    cgroup(f"{container}/memory.current", str(1 * GIB))
    cgroup(f"{pod}/cpu.max", "200000 100000")
    cgroup(f"{pod}/memory.max", str(4 * GIB))
    cgroup(f"{pod}/memory.current", str(2 * GIB))  # includes a sibling container
    assert extractor_module.container_cpu_limit() == 2.0
    assert extractor_module.container_memory_free() == 2 * GIB


def test_cgroup_v1_files_are_read_under_the_controller_mount(cgroup):
    """A cgroup v1 node with a host namespace: each controller is its own
    hierarchy, mounted under its /proc/self/cgroup name, and the process path
    is relative to that mount."""
    cgroup.process(
        "11:memory:/kubepods/pod1/ctr",
        "7:cpu,cpuacct:/kubepods/pod1/ctr",
        "1:name=systemd:/kubepods/pod1/ctr",
    )
    cgroup("cpu,cpuacct/kubepods/pod1/ctr/cpu.cfs_quota_us", "150000")
    cgroup("cpu,cpuacct/kubepods/pod1/ctr/cpu.cfs_period_us", "100000")
    cgroup("cpu,cpuacct/cpu.cfs_quota_us", "-1")  # the mount root: no quota
    cgroup("cpu,cpuacct/cpu.cfs_period_us", "100000")
    cgroup("memory/kubepods/pod1/ctr/memory.limit_in_bytes", str(6 * GIB))
    cgroup("memory/kubepods/pod1/ctr/memory.usage_in_bytes", str(2 * GIB))
    cgroup("memory/memory.limit_in_bytes", "9223372036854771712")  # root: no limit
    cgroup("memory/memory.usage_in_bytes", str(40 * GIB))
    assert extractor_module.container_cpu_limit() == 1.5
    assert extractor_module.container_memory_free() == 4 * GIB


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
            extractor_module._extract_long_concurrently(model, pool, TEXTS, object(), **OPTIONS)
        model.batch_extract = real_batch_extract
        recovered = extractor_module._extract_long_concurrently(
            model, pool, TEXTS, object(), **OPTIONS
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
            extractor_module._extract_long_concurrently, model, pool, TEXTS, object(), **OPTIONS
        )
        future_b = callers.submit(
            extractor_module._extract_long_concurrently,
            model,
            pool,
            other_texts,
            object(),
            **OPTIONS,
        )
        assert future_a.result() == expected_a
        assert future_b.result() == expected_b


def test_repeated_extraction_does_not_grow_the_thread_count(monkeypatch):
    """The process-wide pool is reused: many calls never exceed its size in threads."""
    monkeypatch.setattr(extractor_module, "inference_threads", lambda: 3)
    extractor_module.reset_inference_pool()
    pool = extractor_module._inference_pool()
    model = window_model()
    for _ in range(10):
        extractor_module._extract_long_concurrently(model, pool, TEXTS, object(), **OPTIONS)
    gliner_threads = [t for t in threading.enumerate() if t.name.startswith("gliner")]
    assert 0 < len(gliner_threads) <= 3
    extractor_module.reset_inference_pool()
    time.sleep(0.1)
    assert not [t for t in threading.enumerate() if t.name.startswith("gliner")]
