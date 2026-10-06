"""Remote GLiNER: run the GLiNER demo's model calls on a gliner_worker.

Set ``COGNEE_GLINER_TRANSPORT`` to ``http``, ``grpc`` or ``amqp`` and
``COGNEE_GLINER_ENDPOINT`` to the worker
(https://github.com/topoteretes/gliner_worker), and cognify's GLiNER pipeline
sends its extraction there instead of loading the model in this process: no
torch, no gliner2 and no model download on the cognee side. The worker runs the
same gliner2 calls the local runtime makes (``batch_extract_long`` with the same
windows, overlap policy and label descriptions), so the graph is the one local
extraction builds.

Everything else in the pipeline (chunking, schema resolution, mapping results
to graphs, summaries, storage, embeddings) stays in this process.
"""

from .adapter import RemoteGlinerAdapter, close_shared_workers
from .errors import (
    GlinerRemoteConfigError,
    GlinerWorkerError,
    GlinerWorkerIncompatibleError,
    GlinerWorkerModelMismatchError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from .settings import RemoteGlinerSettings, get_remote_gliner_settings, remote_gliner_configured


def create_remote_adapter(
    settings: RemoteGlinerSettings | None = None,
) -> RemoteGlinerAdapter | None:
    """An adapter for one run when a remote transport is configured, else ``None``."""
    settings = settings or get_remote_gliner_settings()
    if not settings.is_remote:
        return None
    return RemoteGlinerAdapter(settings)


__all__ = [
    "GlinerRemoteConfigError",
    "GlinerWorkerError",
    "GlinerWorkerIncompatibleError",
    "GlinerWorkerModelMismatchError",
    "GlinerWorkerRejectedError",
    "GlinerWorkerRuntimeError",
    "GlinerWorkerUnauthorizedError",
    "GlinerWorkerUnavailableError",
    "RemoteGlinerAdapter",
    "RemoteGlinerSettings",
    "close_shared_workers",
    "create_remote_adapter",
    "get_remote_gliner_settings",
    "remote_gliner_configured",
]
