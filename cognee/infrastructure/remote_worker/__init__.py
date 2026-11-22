"""Parts shared by the clients of cognee's remote model workers.

The GLiNER worker (``cognee.tasks.graph.gliner_demo.remote``) and the embedding
worker (``cognee.infrastructure.databases.vector.embeddings.embedding_worker``)
speak the same RabbitMQ RPC protocol and retry the same way; only their request
bodies, replies and error classes differ. :mod:`.amqp_rpc` needs aio-pika and is
imported only by the transports that use it.
"""

from .retry import RetryPolicy
from .urls import redact_url

__all__ = ["RetryPolicy", "redact_url"]
