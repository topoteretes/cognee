"""An embedding engine's vector size, read without blocking the event loop."""


async def engine_vector_size(embedding_engine) -> int:
    """The engine's vector size, awaited from async code.

    ``get_vector_size()`` is sync. An engine that learns its size from a remote
    worker (``embedding_worker`` without ``EMBEDDING_DIMENSIONS``) can only
    answer it before first contact by blocking the caller's event loop for a
    round trip (or its whole retry budget when the worker is down). Such an
    engine offers ``resolve_vector_size()``, which asks on the running loop;
    every other engine knows its size and answers at once.
    """
    resolve = getattr(embedding_engine, "resolve_vector_size", None)
    if resolve is not None:
        return await resolve()
    return embedding_engine.get_vector_size()
