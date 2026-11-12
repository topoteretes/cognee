"""The on-disk envelope: a header naming the wrapped data key, then the
plaintext sealed in independent AES-256-GCM chunks.

    MAGIC (10 bytes) | header length (uint32 BE) | header (JSON) | chunk 0 | chunk 1 | ...

Each chunk is ``AESGCM(data_key).encrypt(nonce, plaintext, aad)`` — plaintext
bytes plus a 16-byte tag — where the nonce is an 8-byte random prefix from the
header followed by the chunk index, and the AAD is the header bytes, the chunk
index and a final flag. Binding the header stops a header swap (another file's
wrapped key, a changed chunk size); the index stops reordering; the final flag
stops truncation at a chunk boundary, which would otherwise authenticate.

Sealing chunks rather than the whole file keeps memory flat for large inputs
and lets :class:`DecryptingReader` seek: chunk ``i`` starts at a computable
offset, so a seek decrypts one chunk, not everything before it.
"""

import io
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .exceptions import FileDecryptionError

MAGIC = b"COGNEEENC\x01"
FORMAT_VERSION = 1
ALGORITHM = "AES-256-GCM"
TAG_SIZE = 16
NONCE_PREFIX_SIZE = 8
# A header is a few hundred bytes of JSON; anything near this is not ours.
MAX_HEADER_SIZE = 64 * 1024
MAX_CHUNKS = 2**32


def is_envelope(prefix: bytes) -> bool:
    """Whether bytes read from the start of a file open an envelope."""
    return prefix.startswith(MAGIC)


@dataclass(frozen=True)
class EnvelopeHeader:
    chunk_size: int
    provider: str
    key_id: str
    wrapped_key: bytes
    nonce_prefix: bytes
    # The exact JSON bytes on disk: AAD must be bit-identical on both sides, so
    # the parsed header never re-serializes.
    raw: bytes

    @property
    def header_length(self) -> int:
        """Bytes from the start of the file to the first chunk."""
        return len(MAGIC) + 4 + len(self.raw)

    @property
    def sealed_chunk_size(self) -> int:
        return self.chunk_size + TAG_SIZE

    def to_bytes(self) -> bytes:
        return MAGIC + struct.pack(">I", len(self.raw)) + self.raw

    @classmethod
    def build(
        cls, chunk_size: int, provider: str, key_id: str, wrapped_key: bytes
    ) -> "EnvelopeHeader":
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        nonce_prefix = os.urandom(NONCE_PREFIX_SIZE)
        raw = json.dumps(
            {
                "v": FORMAT_VERSION,
                "alg": ALGORITHM,
                "chunk_size": chunk_size,
                "provider": provider,
                "key_id": key_id,
                "wrapped_key": _b64(wrapped_key),
                "nonce_prefix": _b64(nonce_prefix),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(chunk_size, provider, key_id, wrapped_key, nonce_prefix, raw)

    @classmethod
    def parse(cls, stream: BinaryIO) -> "EnvelopeHeader":
        """Read a header from a stream positioned at its start.

        Leaves the stream positioned at the first chunk.
        """
        magic = _read_exact(stream, len(MAGIC))
        if magic != MAGIC:
            raise FileDecryptionError("Not a cognee encrypted file (bad magic).")
        (length,) = struct.unpack(">I", _read_exact(stream, 4))
        if length == 0 or length > MAX_HEADER_SIZE:
            raise FileDecryptionError(f"Encrypted file header length {length} is out of range.")
        raw = _read_exact(stream, length)
        if len(raw) != length:
            raise FileDecryptionError("Encrypted file is truncated inside its header.")
        try:
            fields = json.loads(raw)
            if fields["v"] != FORMAT_VERSION:
                raise FileDecryptionError(
                    f"Encrypted file format version {fields['v']} is not supported "
                    f"(this cognee reads version {FORMAT_VERSION})."
                )
            if fields["alg"] != ALGORITHM:
                raise FileDecryptionError(f"Unsupported cipher '{fields['alg']}'.")
            chunk_size = int(fields["chunk_size"])
            if chunk_size <= 0:
                raise FileDecryptionError("Encrypted file header has a non-positive chunk size.")
            nonce_prefix = _unb64(fields["nonce_prefix"])
            if len(nonce_prefix) != NONCE_PREFIX_SIZE:
                raise FileDecryptionError("Encrypted file header has a malformed nonce prefix.")
            return cls(
                chunk_size=chunk_size,
                provider=str(fields["provider"]),
                key_id=str(fields["key_id"]),
                wrapped_key=_unb64(fields["wrapped_key"]),
                nonce_prefix=nonce_prefix,
                raw=raw,
            )
        except FileDecryptionError:
            raise
        except (KeyError, ValueError, TypeError) as error:
            raise FileDecryptionError(f"Encrypted file header is malformed: {error}") from error


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64

    return base64.b64decode(text, validate=True)


def _nonce(header: EnvelopeHeader, index: int) -> bytes:
    return header.nonce_prefix + struct.pack(">I", index)


def _aad(header: EnvelopeHeader, index: int, final: bool) -> bytes:
    return header.raw + struct.pack(">I?", index, final)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Up to ``size`` bytes, short only at EOF (some streams return less per call)."""
    parts = []
    remaining = size
    while remaining > 0:
        piece = stream.read(remaining)
        if not piece:
            break
        parts.append(piece)
        remaining -= len(piece)
    return b"".join(parts)


def encrypt_stream(
    source: BinaryIO, sink: BinaryIO, data_key: bytes, header: EnvelopeHeader
) -> int:
    """Seal ``source`` into ``sink`` under ``data_key``; returns the plaintext byte count.

    Always writes at least one chunk — an empty file is one empty, final chunk
    (just its tag) — so a reader can tell "empty" from "truncated before the
    first chunk".
    """
    aead = AESGCM(data_key)
    sink.write(header.to_bytes())

    total = 0
    index = 0
    pending = _read_exact(source, header.chunk_size)
    while True:
        # A short read is the end; a full one might be, so peek at the next.
        following = (
            _read_exact(source, header.chunk_size) if len(pending) == header.chunk_size else b""
        )
        final = not following
        sink.write(aead.encrypt(_nonce(header, index), pending, _aad(header, index, final)))
        total += len(pending)
        if final:
            return total
        pending = following
        index += 1
        if index >= MAX_CHUNKS:
            raise ValueError("File too large for the envelope format at this chunk size.")


def plaintext_size(header: EnvelopeHeader, file_size: int) -> int:
    """The unsealed size of an envelope that occupies ``file_size`` bytes."""
    body = file_size - header.header_length
    if body <= 0:
        raise FileDecryptionError("Encrypted file has no sealed body.")
    chunk_count = math.ceil(body / header.sealed_chunk_size)
    size = body - chunk_count * TAG_SIZE
    if size < 0:
        raise FileDecryptionError("Encrypted file body is shorter than its chunk tags.")
    return size


class DecryptingReader(io.BufferedIOBase):
    """A seekable, read-only plaintext view over a sealed stream.

    ``inner`` must be positioned anywhere and be seekable; it is owned by the
    caller (closing the reader closes it, matching the storage ``open``
    contract). Chunks are decrypted on demand and one is cached, so sequential
    reads of any size cost one decryption per chunk and a seek costs at most
    one.
    """

    def __init__(
        self, inner: BinaryIO, header: EnvelopeHeader, data_key: bytes, name: str | None = None
    ) -> None:
        super().__init__()
        self._inner = inner
        self._header = header
        self._aead = AESGCM(data_key)
        self._name = name
        self._body_start = header.header_length

        inner.seek(0, io.SEEK_END)
        self._size = plaintext_size(header, inner.tell())
        body = inner.tell() - self._body_start
        self._chunk_count = math.ceil(body / header.sealed_chunk_size)

        self._pos = 0
        self._cached_index: int | None = None
        self._cached = b""

    # -- metadata ---------------------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    # -- reading ----------------------------------------------------------

    def _load_chunk(self, index: int) -> bytes:
        if index == self._cached_index:
            return self._cached
        self._inner.seek(self._body_start + index * self._header.sealed_chunk_size)
        sealed = _read_exact(self._inner, self._header.sealed_chunk_size)
        final = index == self._chunk_count - 1
        try:
            plaintext = self._aead.decrypt(
                _nonce(self._header, index), sealed, _aad(self._header, index, final)
            )
        except InvalidTag as error:
            raise FileDecryptionError(
                f"Chunk {index} of '{self._name or 'encrypted file'}' failed authentication: "
                "the file is corrupt, truncated, or was written under a different key."
            ) from error
        if not final and len(plaintext) != self._header.chunk_size:
            raise FileDecryptionError(f"Chunk {index} has an unexpected length.")
        self._cached_index = index
        self._cached = plaintext
        return plaintext

    def read(self, size: int | None = -1) -> bytes:
        self._checkClosed()
        if size is None or size < 0:
            size = self._size - self._pos
        out = []
        while size > 0 and self._pos < self._size:
            index, offset = divmod(self._pos, self._header.chunk_size)
            piece = self._load_chunk(index)[offset : offset + size]
            if not piece:
                break
            out.append(piece)
            self._pos += len(piece)
            size -= len(piece)
        return b"".join(out)

    def read1(self, size: int | None = -1) -> bytes:
        return self.read(size)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def readall(self) -> bytes:
        return self.read()

    # -- positioning ------------------------------------------------------

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._pos + offset
        elif whence == io.SEEK_END:
            target = self._size + offset
        else:
            raise ValueError(f"invalid whence ({whence}, should be 0, 1 or 2)")
        if target < 0:
            raise ValueError("negative seek position")
        self._pos = target
        return self._pos

    def tell(self) -> int:
        self._checkClosed()
        return self._pos

    def close(self) -> None:
        if not self.closed:
            try:
                self._inner.close()
            finally:
                super().close()
