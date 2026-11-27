"""Retries for explicitly replay-safe Postgres reads and deterministic upserts.

Never wrap arbitrary transactions with this helper: a connection lost during
commit can leave its outcome unknown. Embedding calls own their retry budget.
"""

import asyncio
import errno
import random
import socket

from cognee.infrastructure.databases.exceptions import EmbeddingException


def is_transient_postgres_error(error: Exception) -> bool:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, EmbeddingException):
            return False
        state = getattr(error, "sqlstate", None)
        if state in {"08001", "08003", "08006", "57P03"}:
            return True
        if state == "08P01" and str(error).strip().lower() == "authentication timed out":
            return True
        if isinstance(error, socket.gaierror):
            return error.errno == socket.EAI_AGAIN
        if isinstance(error, OSError) and error.errno in {
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
        }:
            return True
        error = getattr(error, "orig", None) or error.__cause__
    return False


async def retry_postgres(operation):
    for attempt in range(3):
        try:
            return await operation()
        except Exception as error:
            if attempt == 2 or not is_transient_postgres_error(error):
                raise
            await asyncio.sleep(random.uniform(0.5, 1.0) * (2**attempt))
