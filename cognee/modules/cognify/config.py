import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.exceptions import CogneeConfigurationError
from cognee.shared.data_models import DefaultContentPrediction, SummarizedContent


class CognifyConfig(BaseSettings):
    classification_model: object = DefaultContentPrediction
    summarization_model: object = SummarizedContent
    triplet_embedding: bool = False
    chunks_per_batch: int | None = None
    # Opt-in contradiction detection (issue #3699). Default OFF so the standard
    # cognify pipeline is unchanged. Tunables gate the verdict and the LLM payload.
    contradiction_detection: bool = False
    contradiction_confidence_threshold: float = 0.5
    contradiction_max_facts: int = 500
    # Opt-in audit-grade provenance ledger (env: PROVENANCE_TRACKING). Default
    # OFF so the standard cognify pipeline is unchanged.
    provenance_tracking: bool = False
    # Which implementation fills the extract-and-summarize step of the default
    # cognify pipeline (env: GRAPH_EXTRACTOR). "auto" (default) runs the LLM
    # path when a usable LLM key is configured and the GLiNER demo otherwise;
    # "llm" / "gliner_demo" pin one regardless of credentials. The GLiNER demo
    # installs its runtime on first use (see below) and makes no LLM call.
    graph_extractor: str = "auto"
    # GLiNER's torch is not a cognee dependency: when the demo extractor is
    # resolved and gliner2/torch are missing, CPU-only torch is installed from
    # GLINER_TORCH_INDEX_URL plus the `gliner` extra (gliner_demo/install.py).
    # GLINER_AUTO_INSTALL=false raises KeylessExtractorNotInstalledError instead,
    # for environments installed at build time.
    gliner_auto_install: bool = True
    gliner_torch_index_url: str = "https://download.pytorch.org/whl/cpu"
    # How many GLiNER model batches run at once, sharing one loaded model
    # (env: GLINER_INFERENCE_THREADS). 0 (default) sizes it to the machine:
    # half of torch's thread count, capped by free memory. 1 keeps the
    # single-threaded behaviour. Output is identical at every setting.
    gliner_inference_threads: int = 0
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "triplet_embedding": self.triplet_embedding,
            "chunks_per_batch": self.chunks_per_batch,
            "contradiction_detection": self.contradiction_detection,
            "contradiction_confidence_threshold": self.contradiction_confidence_threshold,
            "contradiction_max_facts": self.contradiction_max_facts,
            "provenance_tracking": self.provenance_tracking,
            "graph_extractor": self.graph_extractor,
            "gliner_auto_install": self.gliner_auto_install,
            "gliner_torch_index_url": self.gliner_torch_index_url,
            "gliner_inference_threads": self.gliner_inference_threads,
        }


@lru_cache
def get_cognify_config():
    return CognifyConfig()


LLM_EXTRACTOR = "llm"
# DEMO: the open-source GLiNER extractor is a demo of cognee's enterprise GLiNER
# extraction, like ``postgres_demo`` is the demo graph backend.
GLINER_DEMO_EXTRACTOR = "gliner_demo"
EXTRACTORS = (LLM_EXTRACTOR, GLINER_DEMO_EXTRACTOR)
EXTRACTOR_ALIASES = {"gliner": GLINER_DEMO_EXTRACTOR}
AUTO_EXTRACTOR = "auto"

GLINER_DEMO_NOTICE = (
    "Extracting the knowledge graph with the GLiNER demo extractor: cognee's open-source "
    "local extraction, free to use with no LLM key. The production-grade GLiNER extraction "
    "(higher accuracy and broader label coverage) is available with a cognee enterprise "
    "licence; write to social@cognee.ai to explore the options."
)
_gliner_demo_notice_logged = False


def _log_gliner_demo_notice_once() -> None:
    """Tell the operator once per process what the demo extractor is and is not."""
    global _gliner_demo_notice_logged
    if _gliner_demo_notice_logged:
        return
    _gliner_demo_notice_logged = True
    from cognee.shared.logging_utils import get_logger

    get_logger("cognify.config").warning(GLINER_DEMO_NOTICE)


class KeylessExtractorNotInstalledError(CogneeConfigurationError):
    """The GLiNER demo extractor is needed, its runtime is missing, and auto-install is off."""

    def __init__(self):
        super().__init__(
            "Cognify would extract the graph with the local GLiNER demo model, but its "
            "runtime (gliner2 + torch) is not installed and GLINER_AUTO_INSTALL is false.",
            "KeylessExtractorNotInstalledError",
            remediation=(
                'Install cognee with the GLiNER extra: pip install "cognee[gliner]" '
                "(or set LLM_API_KEY to extract with an LLM)."
            ),
        )


def resolve_extractor(
    value: str | None, config: CognifyConfig, llm_configured: bool | None = None
) -> str:
    """Resolve the extractor for a cognify run to ``llm`` or ``gliner_demo``.

    The explicit argument wins over ``GRAPH_EXTRACTOR``; the default ``auto``
    picks ``llm`` when a usable LLM key is configured and ``gliner_demo``
    otherwise, so cognee ingests with local models when no credentials are set
    at all. ``llm_configured`` overrides the key check (tests); by default it
    is the inverse of ``keyless_local_defaults_apply()``, which also keeps
    ``llm`` when the preflight is disabled (mocked or deliberately partial
    config). Resolving to the demo extractor logs the enterprise notice once
    per process. It never installs anything: the pipeline entry point awaits
    ``ensure_extractor_runtime`` for that, after its own argument checks.

    This is the ONLY place the extractor setting is read. Callers resolve once,
    up front, and pass the resolved value (or values derived from it) onward —
    no downstream code re-reads the config.
    """
    extractor = (value or config.graph_extractor or AUTO_EXTRACTOR).strip().lower()
    extractor = EXTRACTOR_ALIASES.get(extractor, extractor)
    if extractor == AUTO_EXTRACTOR:
        if llm_configured is None:
            from cognee.modules.preflight import keyless_local_defaults_apply

            llm_configured = not keyless_local_defaults_apply()
        extractor = LLM_EXTRACTOR if llm_configured else GLINER_DEMO_EXTRACTOR
    if extractor not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {extractor!r}; expected one of "
            f"{', '.join((AUTO_EXTRACTOR, *EXTRACTORS))}"
        )
    if extractor == GLINER_DEMO_EXTRACTOR:
        _log_gliner_demo_notice_once()
    return extractor


GLINER_INSTALL_STARTED_EVENT = "GLiNER Runtime Install Started"
GLINER_INSTALL_COMPLETED_EVENT = "GLiNER Runtime Install Completed"
GLINER_INSTALL_FAILED_EVENT = "GLiNER Runtime Install Failed"


def _gliner_install_properties(config: CognifyConfig) -> dict:
    """Environment facts for the install events: versions and platform names only.

    Never paths, URLs, installer output or error messages: those carry usernames,
    hostnames and internal mirrors. A custom index is reported as "custom".
    """
    import platform

    from cognee import __version__ as cognee_version

    default_index = CognifyConfig.model_fields["gliner_torch_index_url"].default
    return {
        "cognee_version": cognee_version,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "torch_index": "pytorch-cpu"
        if config.gliner_torch_index_url == default_index
        else "custom",
    }


async def ensure_extractor_runtime(extractor: str, config: CognifyConfig) -> None:
    """Make the GLiNER runtime importable before a pipeline that uses it starts.

    A no-op for the LLM extractor and when the runtime is present. Otherwise the
    blocking install runs in a worker thread and is awaited, so the event loop keeps
    serving other work while this caller waits for it. Raises
    ``KeylessExtractorNotInstalledError`` when ``GLINER_AUTO_INSTALL`` is off and
    ``GlinerInstallError`` when the install fails.
    """
    import asyncio
    import concurrent.futures

    from cognee.shared.utils import send_telemetry
    from cognee.tasks.graph.gliner_demo.install import (
        GlinerInstallError,
        gliner_runtime_installed,
        install_gliner_runtime,
    )

    if extractor != GLINER_DEMO_EXTRACTOR or gliner_runtime_installed():
        return
    if not config.gliner_auto_install:
        raise KeylessExtractorNotInstalledError()
    properties = _gliner_install_properties(config)
    loop = asyncio.get_running_loop()

    async def started() -> None:
        send_telemetry(GLINER_INSTALL_STARTED_EVENT, "sdk", additional_properties=properties)

    def on_start() -> None:
        # Runs in the install thread, and only in the call that installs (a caller that
        # waited on the lock never gets here). send_telemetry needs the event loop, and
        # Started must land before Completed/Failed: a callback merely scheduled on the
        # loop can run after this coroutine has already resumed (seen on Python 3.14),
        # so wait for the loop to record it. Best effort: telemetry never blocks an
        # install for more than a moment.
        try:
            asyncio.run_coroutine_threadsafe(started(), loop).result(timeout=5)
        except concurrent.futures.TimeoutError:  # a distinct class on Python 3.10
            pass

    try:
        outcome = await asyncio.to_thread(
            install_gliner_runtime, config.gliner_torch_index_url, on_start
        )
    except GlinerInstallError as error:
        send_telemetry(
            GLINER_INSTALL_FAILED_EVENT,
            "sdk",
            additional_properties={
                **properties,
                "failed_step": error.step,
                "installer": error.installer,
                "exception_type": type(error.__cause__ or error).__name__,
            },
        )
        raise
    except Exception as error:
        send_telemetry(
            GLINER_INSTALL_FAILED_EVENT,
            "sdk",
            additional_properties={
                **properties,
                "failed_step": "unexpected",
                "exception_type": type(error).__name__,
            },
        )
        raise
    if not outcome.installed:
        return  # another caller installed it while this one waited on the lock
    send_telemetry(
        GLINER_INSTALL_COMPLETED_EVENT,
        "sdk",
        additional_properties={
            **properties,
            "installer": outcome.installer,
            "installed": outcome.installed,
            "torch_version": outcome.torch_version,
            "duration_seconds": outcome.seconds,
        },
    )


def default_pipeline_needs_llm(extractor: str, config: CognifyConfig) -> bool:
    """True when the default cognify task list contains an LLM task.

    Serves only the early provider preflight in ``add()``/``remember()``,
    which runs before any task list exists: extraction on the ``llm``
    extractor and the opt-in contradiction pass are the LLM tasks of the
    default pipeline. The pipeline-level gate does not use this formula — it
    derives the need from the tasks themselves (``Task.needs_llm`` union, see
    ``pipeline_needs_llm``) and is the authority when the two disagree.
    """
    return extractor == LLM_EXTRACTOR or config.contradiction_detection
