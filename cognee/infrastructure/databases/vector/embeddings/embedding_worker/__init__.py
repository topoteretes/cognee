"""Client parts for cognee's embedding worker (EMBEDDING_PROVIDER=embedding_worker).

The engine is :class:`..EmbeddingWorkerEmbeddingEngine.EmbeddingWorkerEmbeddingEngine`;
this package holds its settings, typed errors, wire contract, RabbitMQ transport
and the per-event-loop transport cache. Importing it does not import aio-pika.
"""

from .errors import (
    EmbeddingWorkerConfigError,
    EmbeddingWorkerDimensionsMismatchError,
    EmbeddingWorkerError,
    EmbeddingWorkerInputTooLongError,
    EmbeddingWorkerModelMismatchError,
    EmbeddingWorkerRejectedError,
    EmbeddingWorkerRuntimeError,
    EmbeddingWorkerTimeoutError,
    EmbeddingWorkerUnauthorizedError,
    EmbeddingWorkerUnavailableError,
)
from .settings import EmbeddingWorkerSettings, get_embedding_worker_settings
from .shared import close_embedding_worker_transports

__all__ = [
    "EmbeddingWorkerConfigError",
    "EmbeddingWorkerDimensionsMismatchError",
    "EmbeddingWorkerError",
    "EmbeddingWorkerInputTooLongError",
    "EmbeddingWorkerModelMismatchError",
    "EmbeddingWorkerRejectedError",
    "EmbeddingWorkerRuntimeError",
    "EmbeddingWorkerSettings",
    "EmbeddingWorkerTimeoutError",
    "EmbeddingWorkerUnauthorizedError",
    "EmbeddingWorkerUnavailableError",
    "close_embedding_worker_transports",
    "get_embedding_worker_settings",
]
