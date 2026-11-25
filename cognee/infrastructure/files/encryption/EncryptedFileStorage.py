import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from tempfile import SpooledTemporaryFile
from typing import BinaryIO, cast

from cognee.infrastructure.files.storage.storage import Storage
from cognee.infrastructure.utils.run_async import run_async

from .envelope import MAGIC, DecryptingReader, EnvelopeHeader, encrypt_stream, is_envelope
from .envelope import plaintext_size as envelope_plaintext_size
from .key_providers import KeyProvider, unwrap_data_key_cached

# Ciphertext is staged here before the inner store streams it out. Under this
# size it stays in memory; above, it spools to a temp file — ciphertext, so a
# spill is harmless.
SPOOL_MAX_MEMORY = 8 * 1024 * 1024


class EncryptedFileStorage(Storage):
    """Seal every object on the way into ``inner`` and unseal it on the way out.

    Transparent to callers of the :class:`Storage` protocol: ``store`` takes
    the same plaintext, ``open`` yields plaintext (binary or text mode), sizes
    are plaintext sizes. Objects already in ``inner`` that are not envelopes —
    written before encryption was enabled — pass through unchanged, so turning
    the layer on never makes existing data unreadable; they are sealed the
    next time they are written.

    Only reading and writing whole objects is supported: ``open`` refuses
    write modes, because the chunked format cannot be appended to in place.
    """

    def __init__(self, inner: Storage, provider: KeyProvider, chunk_size: int) -> None:
        self.inner = inner
        self.provider = provider
        self.chunk_size = chunk_size

    @property
    def storage_path(self) -> str:
        return self.inner.storage_path

    # -- writing ----------------------------------------------------------

    def _seal(self, source: BinaryIO) -> BinaryIO:
        data_key = self.provider.generate_data_key()
        header = EnvelopeHeader.build(
            chunk_size=self.chunk_size,
            provider=self.provider.name,
            key_id=data_key.key_id,
            wrapped_key=data_key.wrapped,
        )
        # SpooledTemporaryFile in "w+b" mode is a binary stream, but typeshed
        # types it as IO[bytes] rather than BinaryIO; the cast records that.
        spool = cast(
            BinaryIO,
            SpooledTemporaryFile(  # noqa: SIM115 - handed to the caller, closed in store()
                max_size=SPOOL_MAX_MEMORY, mode="w+b"
            ),
        )
        try:
            encrypt_stream(source, spool, data_key.plaintext, header)
        except BaseException:
            spool.close()
            raise
        spool.seek(0)
        return spool

    async def store(self, file_path: str, data: BinaryIO | str, overwrite: bool = False) -> str:
        if isinstance(data, str):
            # Both backends write a str as UTF-8 with no newline translation;
            # the sealed bytes are exactly those.
            source: BinaryIO = io.BytesIO(data.encode("utf-8"))
        elif isinstance(data, (bytes, bytearray, memoryview)):
            source = io.BytesIO(bytes(data))
        else:
            data.seek(0)
            source = data

        # Key generation may be a KMS round trip and sealing is CPU work; neither
        # belongs on the event loop.
        spool = await run_async(self._seal, source)
        try:
            return await self.inner.store(file_path, spool, overwrite)
        finally:
            spool.close()

    # -- reading ----------------------------------------------------------

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "rb", *args, **kwargs) -> AsyncGenerator:
        if any(flag in mode for flag in "wax+"):
            raise ValueError("EncryptedFileStorage.open is read-only; write with store().")
        text_mode = "b" not in mode
        encoding = kwargs.pop("encoding", None)
        errors = kwargs.pop("errors", None)
        newline = kwargs.pop("newline", None)

        sealed = False
        async with self.inner.open(file_path, "rb", *args, **kwargs) as raw:
            prefix = raw.read(len(MAGIC))
            raw.seek(0)
            sealed = is_envelope(prefix)
            if sealed:
                header = EnvelopeHeader.parse(raw)
                data_key = await run_async(
                    unwrap_data_key_cached, self.provider, header.wrapped_key, header.key_id
                )
                reader = DecryptingReader(raw, header, data_key, name=getattr(raw, "name", None))
                if text_mode:
                    text = io.TextIOWrapper(
                        reader, encoding=encoding or "utf-8", errors=errors, newline=newline
                    )
                    try:
                        yield text
                    finally:
                        # The inner context closes the underlying stream; detach so
                        # the wrapper's own close does not race it.
                        try:
                            text.detach()
                        except ValueError:
                            pass
                else:
                    yield reader
            elif not text_mode:
                yield raw

        if not sealed and text_mode:
            # A plaintext object asked for in text mode: hand it to the backend
            # exactly as before encryption existed, with the caller's own
            # encoding and newline handling.
            if encoding is not None:
                kwargs["encoding"] = encoding
            if errors is not None:
                kwargs["errors"] = errors
            if newline is not None:
                kwargs["newline"] = newline
            async with self.inner.open(file_path, mode, *args, **kwargs) as plain:
                yield plain

    async def get_size(self, file_path: str) -> int:
        """Plaintext size: what a reader of ``open`` will see, not the bytes on disk."""
        total = await self.inner.get_size(file_path)
        if total < len(MAGIC):
            return total
        async with self.inner.open(file_path, "rb") as raw:
            prefix = raw.read(len(MAGIC))
            if not is_envelope(prefix):
                return total
            raw.seek(0)
            header = EnvelopeHeader.parse(raw)
        return envelope_plaintext_size(header, total)

    async def is_sealed(self, file_path: str) -> bool:
        """Whether the stored object is an envelope (as opposed to legacy plaintext)."""
        if not await self.inner.file_exists(file_path):
            return False
        async with self.inner.open(file_path, "rb") as raw:
            return is_envelope(raw.read(len(MAGIC)))

    # -- pass-through -----------------------------------------------------

    async def file_exists(self, file_path: str) -> bool:
        return await self.inner.file_exists(file_path)

    async def is_file(self, file_path: str) -> bool:
        return await self.inner.is_file(file_path)

    async def ensure_directory_exists(self, directory_path: str = "") -> None:
        return await self.inner.ensure_directory_exists(directory_path)

    async def copy_file(self, source_file_path: str, destination_file_path: str) -> str:
        # Sealed bytes copy as they are: the copy carries the same wrapped key.
        return await self.inner.copy_file(source_file_path, destination_file_path)

    async def remove(self, file_path: str) -> None:
        return await self.inner.remove(file_path)

    async def list_files(self, directory_path: str, recursive: bool = False) -> list[str]:
        return await self.inner.list_files(directory_path, recursive)

    async def remove_all(self, root_path: str | None = None) -> None:
        return await self.inner.remove_all(root_path)
