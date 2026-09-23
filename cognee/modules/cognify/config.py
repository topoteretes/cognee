import importlib.util
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
    # requires the `gliner` extra and makes no LLM call.
    graph_extractor: str = "auto"
    # Runtime for the GLiNER demo extractor (env: GLINER_BACKEND): "torch" (default)
    # runs gliner2 as shipped; "onnx" runs it on ONNX Runtime with no torch,
    # transformers or gliner2 installed, from a one-time export
    # (python -m cognee.tasks.graph.gliner_demo.onnx.export).
    gliner_backend: str = "torch"
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
            "gliner_backend": self.gliner_backend,
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
    """No LLM key is configured and the local extractor's package is missing."""

    def __init__(self):
        super().__init__(
            "No LLM API key is configured, so cognify would extract the graph with the "
            "local GLiNER demo model, but the `gliner2` package is not installed. Either "
            'install it with: pip install "cognee[gliner]" or set LLM_API_KEY to extract '
            "with an LLM.",
            "KeylessExtractorNotInstalledError",
        )


def _gliner_runtime_installed() -> bool:
    """gliner2 for the torch backend; nothing extra for GLINER_BACKEND=onnx."""
    from cognee.tasks.graph.gliner_demo.extractor import gliner_runtime_installed

    return gliner_runtime_installed()


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
    per process.

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
        if extractor == GLINER_DEMO_EXTRACTOR and not _gliner_runtime_installed():
            raise KeylessExtractorNotInstalledError()
    if extractor not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {extractor!r}; expected one of "
            f"{', '.join((AUTO_EXTRACTOR, *EXTRACTORS))}"
        )
    if extractor == GLINER_DEMO_EXTRACTOR:
        _log_gliner_demo_notice_once()
    return extractor


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
