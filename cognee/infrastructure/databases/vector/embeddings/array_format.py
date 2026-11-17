"""Array output for stores accepting float32, without changing list-based clients."""

from contextvars import ContextVar

import numpy as np
from numpy.typing import NDArray

arrays_requested: ContextVar[bool] = ContextVar("embedding_arrays_requested", default=False)


def native_embedding(vector: NDArray) -> NDArray | list[float]:
    """Keep a provider's native array when the consuming store accepts arrays."""
    return vector if arrays_requested.get() else vector.tolist()


async def embed_float32(engine, texts: list[str]) -> list[NDArray[np.float32]]:
    """Embed once, preserving native/base64 arrays through the provider and gateway.

    JSON providers necessarily produce lists; convert those once, after any recursive
    splitting/pooling, so float64 pooling keeps its existing numeric semantics.
    ContextVar scoping keeps simultaneous list consumers independent, including errors
    and cancellation. Usage accounting stays in EmbeddingGateway.embed_text.
    """
    token = arrays_requested.set(True)
    try:
        vectors = await engine.embed_text(texts)
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]
    finally:
        arrays_requested.reset(token)
