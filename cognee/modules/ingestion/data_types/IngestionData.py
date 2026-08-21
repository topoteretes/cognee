from typing import Protocol, BinaryIO, Union


class IngestionData(Protocol):
    data: Union[str, BinaryIO] = None

    def get_data(self):
        raise NotImplementedError("Subclasses must implement get_data()")

    def get_identifier(self):
        raise NotImplementedError("Subclasses must implement get_identifier()")

    def get_metadata(self):
        raise NotImplementedError("Subclasses must implement get_metadata()")

    async def aget_identifier(self):
        """Async ``get_identifier``. Prefer this from coroutines.

        The sync variants bridge to async through ``run_sync``, which starts a
        thread and ``join()``s it on the calling thread — from a coroutine that
        parks the event loop for the whole read. Async callers must use these.
        """
        raise NotImplementedError("Subclasses must implement aget_identifier()")

    async def aget_metadata(self):
        """Async ``get_metadata``. See :meth:`aget_identifier`."""
        raise NotImplementedError("Subclasses must implement aget_metadata()")
