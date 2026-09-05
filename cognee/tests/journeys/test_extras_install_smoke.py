"""Journey 15: every install extra resolves, installs, and imports.

Two tiers:

* **Resolution** (always on, seconds): ``uv pip compile`` proves each extra
  alone and all extras together have a consistent solution for every supported
  Python version. If the full set fails, the pairs are bisected so the failure
  names the two extras that conflict, e.g. ``docling-full`` vs ``codegraph``.

* **Install smoke** (opt-in, ``COGNEE_JOURNEY_EXTRAS=1``): build the wheel once,
  then for each extra create an empty venv, install ``cognee[extra]`` from the
  wheel, and run ``extras_probe.py`` inside it: ``import cognee`` still works,
  every top-level module of the extra's direct requirements imports, and the
  cognee modules the extra enables import. Select a subset with
  ``COGNEE_EXTRAS=neo4j,redis``; ``COGNEE_EXTRAS_SKIP`` defaults to extras that
  compile from source (``llama-cpp``).
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")

# Extras that compile native code from source (needs a toolchain, ~10 min).
DEFAULT_SKIP = {"llama-cpp"}

# The cognee modules an extra is meant to unlock. Import failure here means the
# extra installs but the feature it advertises does not load.
COGNEE_MODULES_BY_EXTRA: dict[str, list[str]] = {
    "neo4j": ["cognee.infrastructure.databases.graph.neo4j_driver.adapter"],
    "neptune": ["cognee.infrastructure.databases.graph.neptune_driver.adapter"],
    "postgres": ["cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter"],
    "postgres-binary": ["cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter"],
    "turso": ["cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter"],
    "fastembed": ["cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine"],
    "mistral": [
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter"
    ],
    "anthropic": [
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter"
    ],
    "azure": [
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.azure_openai.adapter"
    ],
    "llama-cpp": [
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llama_cpp.adapter"
    ],
    "huggingface": ["cognee.infrastructure.llm.tokenizer.HuggingFace.adapter"],
    "ollama": ["cognee.infrastructure.llm.tokenizer.HuggingFace.adapter"],
    "codegraph": ["cognee.infrastructure.llm.tokenizer.HuggingFace.adapter"],
    "docling": ["cognee.infrastructure.loaders.external.docling_loader"],
    "docling-full": ["cognee.infrastructure.loaders.external.docling_loader"],
    "docs": ["cognee.infrastructure.loaders.external.unstructured_loader"],
    "dlt": ["cognee.tasks.ingestion.create_dlt_source"],
    "gmail": ["cognee.tasks.ingestion.create_dlt_source"],
    "aws": ["cognee.infrastructure.files.storage.S3FileStorage"],
    "redis": ["cognee.infrastructure.databases.cache.redis.RedisAdapter"],
    "tracing": ["cognee.modules.observability.metrics"],
    "graphiti": ["cognee.tasks.temporal_awareness.build_graph_with_temporal_awareness"],
    "llama-index": ["cognee.tasks.ingestion.transform_data"],
    "scraping": ["cognee.tasks.web_scraper.default_url_crawler"],
    "deepeval": ["cognee.eval_framework.evaluation.deep_eval_adapter"],
    "rapidocr": ["cognee.infrastructure.loaders.core.image_loader"],
    "evals": ["cognee.eval_framework.metrics_dashboard"],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extras() -> dict[str, list[str]]:
    if tomllib is None:
        pytest.skip("tomllib unavailable (Python 3.10 without tomli)")
    with PYPROJECT.open("rb") as handle:
        return dict(tomllib.load(handle)["project"]["optional-dependencies"])


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def _resolve(extras: list[str], python_version: str, out: Path) -> subprocess.CompletedProcess:
    cmd = [
        "uv",
        "pip",
        "compile",
        str(PYPROJECT),
        "--quiet",
        "--python-version",
        python_version,
        "-o",
        str(out),
    ]
    for extra in extras:
        cmd += ["--extra", extra]
    return _run(cmd, cwd=REPO_ROOT)


def _venv_bin(venv: Path, name: str) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _selected_extras() -> list[str]:
    all_extras = [e for e in _extras() if e != "api" or _extras()["api"]]  # 'api' is empty
    wanted = os.getenv("COGNEE_EXTRAS", "all").strip()
    skip = {
        s.strip()
        for s in os.getenv("COGNEE_EXTRAS_SKIP", ",".join(DEFAULT_SKIP)).split(",")
        if s.strip()
    }
    chosen = (
        all_extras if wanted in ("", "all") else [e.strip() for e in wanted.split(",") if e.strip()]
    )
    unknown = sorted(set(chosen) - set(all_extras))
    assert not unknown, f"COGNEE_EXTRAS names unknown extras: {unknown}"
    return [e for e in chosen if e not in skip]


# ---------------------------------------------------------------------------
# tier 1: resolution (always on)
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.journey,
    pytest.mark.skipif(
        shutil.which("uv") is None, reason="uv is required to resolve and install extras"
    ),
]


@pytest.mark.parametrize("python_version", SUPPORTED_PYTHONS)
def test_all_extras_resolve_together(tmp_path, python_version):
    """One consistent solution for every extra at once means no pair conflicts."""
    extras = sorted(_extras())
    result = _resolve(extras, python_version, tmp_path / "all.txt")
    if result.returncode == 0:
        return

    # Name the culprits: which pairs cannot coexist on this Python?
    conflicting = []
    for a, b in itertools.combinations(extras, 2):
        pair = _resolve([a, b], python_version, tmp_path / f"{a}-{b}.txt")
        if pair.returncode != 0:
            conflicting.append(f"{a} + {b}")
    alone = [
        e for e in extras if _resolve([e], python_version, tmp_path / f"{e}.txt").returncode != 0
    ]
    pytest.fail(
        f"cognee extras do not resolve together on Python {python_version}.\n"
        + (f"  extras that fail alone: {alone}\n" if alone else "")
        + (f"  conflicting pairs: {conflicting}\n" if conflicting else "")
        + "uv output:\n"
        + result.stderr[-3000:]
    )


def test_each_extra_resolves_alone(tmp_path):
    """Per-extra check on the running interpreter's version with a readable failure."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    failures = {}
    for extra in sorted(_extras()):
        result = _resolve([extra], version, tmp_path / f"{extra}.txt")
        if result.returncode != 0:
            failures[extra] = result.stderr[-1500:]
    assert not failures, (
        "extras that do not resolve on Python " + version + ":\n" + json.dumps(failures, indent=2)
    )


# ---------------------------------------------------------------------------
# tier 2: install + import smoke (opt-in)
# ---------------------------------------------------------------------------

_INSTALL_SMOKE = pytest.mark.skipif(
    os.getenv("COGNEE_JOURNEY_EXTRAS") != "1",
    reason="set COGNEE_JOURNEY_EXTRAS=1 to build the wheel and install every extra in its own venv",
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    dist = tmp_path_factory.mktemp("extras-dist")
    build = _run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=REPO_ROOT)
    assert build.returncode == 0, f"uv build failed:\n{build.stdout}\n{build.stderr}"
    wheels = sorted(dist.glob("cognee-*.whl"))
    assert wheels, f"no wheel produced in {dist}"
    return wheels[-1]


def _extra_ids() -> list[str]:
    # Evaluated at collection so the parametrize ids are the extra names.
    try:
        return _selected_extras()
    except Exception:
        return []


@_INSTALL_SMOKE
@pytest.mark.parametrize("extra", _extra_ids())
def test_extra_installs_and_imports(extra, wheel, tmp_path_factory):
    requirements = _extras()[extra]
    work = tmp_path_factory.mktemp(f"extra-{extra}")
    venv = work / "venv"

    create = _run(["uv", "venv", str(venv), "--python", sys.executable])
    assert create.returncode == 0, f"uv venv failed:\n{create.stderr[-2000:]}"
    python = _venv_bin(venv, "python")

    spec = (
        f"cognee[{extra}] @ {wheel.resolve().as_uri()}"
        if requirements
        else f"cognee @ {wheel.resolve().as_uri()}"
    )
    install = _run(["uv", "pip", "install", "--python", str(python), spec])
    assert install.returncode == 0, (
        f"`pip install cognee[{extra}]` fails for a user:\n{install.stderr[-4000:]}"
    )

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LLM_", "EMBEDDING_", "COGNEE_"))
        and k not in ("PYTHONPATH", "VIRTUAL_ENV")
    }
    env.update(
        {
            "DATA_ROOT_DIRECTORY": str(work / "data"),
            "SYSTEM_ROOT_DIRECTORY": str(work / "system"),
            "TELEMETRY_DISABLED": "1",
            "LLM_API_KEY": "smoke-key",
            "COGNEE_SKIP_PREFLIGHT": "1",
        }
    )
    probe = _run(
        [
            str(python),
            str(HERE / "extras_probe.py"),
            "--extra",
            extra,
            "--requirements",
            json.dumps(requirements),
            "--cognee-modules",
            json.dumps(COGNEE_MODULES_BY_EXTRA.get(extra, [])),
        ],
        env=env,
        cwd=work,
    )
    lines = [line for line in probe.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, (
        f"probe printed no report for [{extra}]:\n{probe.stdout[-2000:]}\n{probe.stderr[-3000:]}"
    )
    report = json.loads(lines[-1])

    assert report["ok"], (
        f"cognee[{extra}] installs but does not import cleanly.\n"
        f"  failed imports: {json.dumps({k: v['error'] for k, v in report['failed'].items()}, indent=2)}\n"
        f"  requirements with no installed distribution: {report['missing_dists']}\n"
        f"  stderr tail:\n{probe.stderr[-2000:]}"
    )
    assert "cognee" in report["imported"]
