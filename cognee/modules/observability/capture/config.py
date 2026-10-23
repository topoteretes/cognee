"""Configuration for eval capture (SDK-529).

Fields are named after the full environment variables (repo convention — no
``env_prefix``). The ``COGNEE_`` prefix follows ``COGNEE_TRACING_ENABLED``;
unprefixed ``CAPTURE_*`` would be ambiguous.

The base config is read lazily inside the validator: ``cognee/base_config.py``
imports the observability package, so a module-level ``get_base_config`` import
here would create a partially-initialized-module cycle, and a class-time default
would freeze ``DATA_ROOT_DIRECTORY`` and break monkeypatching in tests.
"""

import os
from functools import lru_cache

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

# A leaf module (no cognee imports), so unlike base_config it is safe here.
from cognee.root_dir import ensure_absolute_path

# Where records go. ``storage``: gzip'd JSONL under COGNEE_CAPTURE_DIR. ``proxy``:
# the analytics proxy product telemetry already uses (see proxy_sink.py).
# ``both``: the two together.
CAPTURE_SINK_MODES = ("storage", "proxy", "both")


class CaptureConfig(BaseSettings):
    """Knobs for the non-blocking eval capture hook.

    Read once per process by ``hook._ensure_initialized()``; never on the emit
    path. Directory defaults to ``<data_root>/capture`` (not ``evals``, which
    collides with ``cognee/eval_framework`` artifacts).

    Enabling capture writes derived user content to that directory: the raw
    per-chunk graphs (extracted entity names, descriptions, relationships),
    the names looked up against the ontology, and retrieval candidate ids —
    filed per dataset and run. It inherits the directory's permissions and is
    not removed when a dataset is forgotten, so a deployment that turns it on
    owns the same retention and access-control obligations for it as for the
    data root. ``COGNEE_CAPTURE_RETRIEVAL_SAMPLE_RATE`` samples only the
    ``retrieval.*`` payloads; the extraction kinds are always captured in full.

    ``COGNEE_CAPTURE_SINK`` picks where records go. ``storage`` (the default)
    keeps everything on the machine. ``proxy`` ships them through the analytics
    proxy product telemetry uses, metadata only: the content kinds leave as
    counts and type histograms, model settings that are paths as
    ``local_path``, ids as they are — unless ``COGNEE_CAPTURE_PROXY_CONTENT``
    is set, which sends the raw chunk graphs and ontology lookups too (derived
    user content leaving the machine; the proxy must be configured to accept
    it). The proxy sink also obeys the telemetry kill switches
    (``TELEMETRY_DISABLED``, ``ENV=test|dev``).
    """

    cognee_capture_enabled: bool = False
    cognee_capture_dir: str | None = None
    cognee_capture_sink: str = "storage"
    # None: the telemetry proxy's capture route (``proxy_sink.default_proxy_url``).
    cognee_capture_proxy_url: str | None = None
    cognee_capture_proxy_content: bool = False
    # Upper bound on one POST to the proxy, like TELEMETRY_REQUEST_TIMEOUT.
    cognee_capture_proxy_timeout_s: float = 5.0
    # Must not be a fraction of cognify's default ``chunks_per_batch`` (2000):
    # the per-chunk emit points run over a whole task batch, and a burst past
    # this bound is dropped newest-first. 2048 covers the default batch with the
    # flusher draining alongside (``capture_chunk_graphs`` yields every 64).
    cognee_capture_queue_size: int = 2048
    cognee_capture_batch_size: int = 64
    cognee_capture_flush_interval_s: float = 2.0
    cognee_capture_retrieval_sample_rate: float = 1.0
    # Upper bound on one sink write; a wedged sink (S3 under partition) must not
    # pin the flusher and every later drain().
    cognee_capture_sink_timeout_s: float = 30.0
    # Default budget for drain() when the caller passes no timeout — the wait the
    # pipeline wiring adds at every run completion while capture is on. Lower it
    # when that tail latency matters more than losing a slow sink's last batch
    # to the atexit hook.
    cognee_capture_drain_timeout_s: float = 5.0

    model_config = SettingsConfigDict(extra="allow")

    @pydantic.model_validator(mode="after")
    def fill_derived(self):
        if not 0.0 <= self.cognee_capture_retrieval_sample_rate <= 1.0:
            raise ValueError(
                "COGNEE_CAPTURE_RETRIEVAL_SAMPLE_RATE must be in [0, 1], "
                f"got {self.cognee_capture_retrieval_sample_rate}"
            )
        if self.cognee_capture_sink_timeout_s <= 0:
            raise ValueError(
                "COGNEE_CAPTURE_SINK_TIMEOUT_S must be positive, "
                f"got {self.cognee_capture_sink_timeout_s}"
            )
        if self.cognee_capture_drain_timeout_s <= 0:
            raise ValueError(
                "COGNEE_CAPTURE_DRAIN_TIMEOUT_S must be positive, "
                f"got {self.cognee_capture_drain_timeout_s}"
            )
        # A batch size of 0 would make every flush pop nothing — and a flusher
        # that pops nothing without suspending monopolises its event loop.
        if self.cognee_capture_queue_size < 1:
            raise ValueError(
                f"COGNEE_CAPTURE_QUEUE_SIZE must be at least 1, got {self.cognee_capture_queue_size}"
            )
        if self.cognee_capture_batch_size < 1:
            raise ValueError(
                f"COGNEE_CAPTURE_BATCH_SIZE must be at least 1, got {self.cognee_capture_batch_size}"
            )
        if self.cognee_capture_flush_interval_s <= 0:
            raise ValueError(
                "COGNEE_CAPTURE_FLUSH_INTERVAL_S must be positive, "
                f"got {self.cognee_capture_flush_interval_s}"
            )

        sink = (self.cognee_capture_sink or "").strip().lower()
        if sink not in CAPTURE_SINK_MODES:
            raise ValueError(
                f"COGNEE_CAPTURE_SINK must be one of {', '.join(CAPTURE_SINK_MODES)}, "
                f"got {self.cognee_capture_sink!r}"
            )
        self.cognee_capture_sink = sink
        if self.cognee_capture_proxy_timeout_s <= 0:
            raise ValueError(
                "COGNEE_CAPTURE_PROXY_TIMEOUT_S must be positive, "
                f"got {self.cognee_capture_proxy_timeout_s}"
            )
        if self.cognee_capture_proxy_url is not None:
            proxy_url = self.cognee_capture_proxy_url.strip()
            if not proxy_url.startswith(("http://", "https://")):
                raise ValueError(
                    "COGNEE_CAPTURE_PROXY_URL must be an http(s) URL, "
                    f"got {self.cognee_capture_proxy_url!r}"
                )
            self.cognee_capture_proxy_url = proxy_url

        if not self.cognee_capture_dir:
            # Lazy on purpose — see the module docstring.
            from cognee.base_config import get_base_config

            base_config = get_base_config()
            self.cognee_capture_dir = os.path.join(base_config.data_root_directory, "capture")

        # Same rule as every other storage root (BaseConfig.validate_paths): a
        # relative value would resolve against the process CWD and scatter
        # gzip'd chunk graphs — verbatim entity names — wherever the server or
        # CLI happened to start, the repo working tree included. Loud at
        # startup instead; s3:// is absolute by definition.
        self.cognee_capture_dir = ensure_absolute_path(self.cognee_capture_dir)

        return self

    def to_dict(self) -> dict:
        """Return the configuration as a dictionary."""
        return {
            "cognee_capture_enabled": self.cognee_capture_enabled,
            "cognee_capture_dir": self.cognee_capture_dir,
            "cognee_capture_sink": self.cognee_capture_sink,
            "cognee_capture_proxy_url": self.cognee_capture_proxy_url,
            "cognee_capture_proxy_content": self.cognee_capture_proxy_content,
            "cognee_capture_proxy_timeout_s": self.cognee_capture_proxy_timeout_s,
            "cognee_capture_queue_size": self.cognee_capture_queue_size,
            "cognee_capture_batch_size": self.cognee_capture_batch_size,
            "cognee_capture_flush_interval_s": self.cognee_capture_flush_interval_s,
            "cognee_capture_retrieval_sample_rate": self.cognee_capture_retrieval_sample_rate,
            "cognee_capture_sink_timeout_s": self.cognee_capture_sink_timeout_s,
            "cognee_capture_drain_timeout_s": self.cognee_capture_drain_timeout_s,
        }


@lru_cache
def get_capture_config() -> CaptureConfig:
    """Return the cached capture configuration.

    Tests that change the environment call ``get_capture_config.cache_clear()``
    (``hook._reset_for_tests()`` does it too).
    """
    return CaptureConfig()
