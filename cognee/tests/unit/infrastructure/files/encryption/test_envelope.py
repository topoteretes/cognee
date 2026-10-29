"""The sealed format: round trips, random access, and what must fail."""

import io
import os

import pytest

from cognee.infrastructure.files.encryption import (
    DecryptingReader,
    EnvelopeHeader,
    FileDecryptionError,
    encrypt_stream,
    is_envelope,
)
from cognee.infrastructure.files.encryption.envelope import MAGIC, TAG_SIZE, plaintext_size

KEY = os.urandom(32)


def seal(plaintext: bytes, chunk_size: int = 16) -> tuple[bytes, EnvelopeHeader]:
    header = EnvelopeHeader.build(chunk_size, "test", "k1", b"wrapped")
    sink = io.BytesIO()
    written = encrypt_stream(io.BytesIO(plaintext), sink, KEY, header)
    assert written == len(plaintext)
    return sink.getvalue(), header


def reader(sealed: bytes) -> DecryptingReader:
    stream = io.BytesIO(sealed)
    header = EnvelopeHeader.parse(stream)
    return DecryptingReader(stream, header, KEY, name="file:///x.txt")


@pytest.mark.parametrize(
    "plaintext",
    [
        b"",
        b"a",
        b"exactly sixteen!",  # one full chunk, no remainder
        b"exactly sixteen!" * 3,  # several full chunks, no remainder
        b"exactly sixteen!" * 3 + b"tail",
        os.urandom(1000),
    ],
)
def test_round_trip(plaintext):
    sealed, header = seal(plaintext)

    assert is_envelope(sealed)
    assert sealed.startswith(MAGIC)
    if len(plaintext) >= 8:  # a lone byte can occur in the JSON header by chance
        assert plaintext not in sealed
    assert plaintext_size(header, len(sealed)) == len(plaintext)

    with reader(sealed) as view:
        assert view.size == len(plaintext)
        assert view.read() == plaintext


def test_header_round_trips_the_exact_bytes():
    _, header = seal(b"x")
    parsed = EnvelopeHeader.parse(io.BytesIO(header.to_bytes()))

    assert parsed == header
    assert parsed.raw == header.raw  # the AAD on both sides


def test_small_reads_cross_chunk_boundaries():
    plaintext = bytes(range(256)) * 2
    sealed, _ = seal(plaintext, chunk_size=30)

    with reader(sealed) as view:
        out = b""
        while piece := view.read(7):
            out += piece
    assert out == plaintext


def test_seek_and_tell_follow_plaintext_offsets():
    plaintext = b"0123456789" * 10
    sealed, _ = seal(plaintext, chunk_size=16)

    with reader(sealed) as view:
        assert view.seek(37) == 37
        assert view.read(5) == plaintext[37:42]
        assert view.tell() == 42
        view.seek(-3, io.SEEK_END)
        assert view.read() == plaintext[-3:]
        view.seek(0)
        view.seek(10, io.SEEK_CUR)
        assert view.read(2) == plaintext[10:12]
        # Past the end reads nothing and stays put.
        view.seek(10_000)
        assert view.read() == b""


def test_text_wrapper_over_reader():
    plaintext = "héllo\nwörld\n".encode()
    sealed, _ = seal(plaintext, chunk_size=5)

    text = io.TextIOWrapper(reader(sealed), encoding="utf-8")
    assert text.read() == "héllo\nwörld\n"


def test_truncated_file_fails_authentication():
    plaintext = b"exactly sixteen!" * 4
    sealed, _ = seal(plaintext)

    # Drop the whole last chunk: boundary truncation, the case a per-chunk
    # scheme without a final flag would accept.
    truncated = sealed[: -(16 + TAG_SIZE)]
    with pytest.raises(FileDecryptionError, match="authentication"), reader(truncated) as view:
        view.seek(-1, io.SEEK_END)
        view.read()


def test_flipped_byte_fails_authentication():
    sealed, _ = seal(b"exactly sixteen!" * 2)
    damaged = bytearray(sealed)
    damaged[-5] ^= 0x01
    with pytest.raises(FileDecryptionError, match="authentication"):
        reader(bytes(damaged)).read()


def test_reordered_chunks_fail_authentication():
    sealed, header = seal(b"AAAAAAAAAAAAAAAA" + b"BBBBBBBBBBBBBBBB" + b"CC")
    start = header.header_length
    size = header.sealed_chunk_size
    body = sealed[start:]
    swapped = body[size : 2 * size] + body[:size] + body[2 * size :]
    with pytest.raises(FileDecryptionError, match="authentication"):
        reader(sealed[:start] + swapped).read()


def test_header_swap_fails_authentication():
    sealed_a, header_a = seal(b"one file's content")
    _, header_b = seal(b"another file, same key")
    forged = header_b.to_bytes() + sealed_a[header_a.header_length :]
    with pytest.raises(FileDecryptionError, match="authentication"):
        reader(forged).read()


def test_wrong_key_fails_authentication():
    sealed, _ = seal(b"secret")
    stream = io.BytesIO(sealed)
    header = EnvelopeHeader.parse(stream)
    with pytest.raises(FileDecryptionError):
        DecryptingReader(stream, header, os.urandom(32)).read()


def test_malformed_headers_are_refused():
    with pytest.raises(FileDecryptionError, match="magic"):
        EnvelopeHeader.parse(io.BytesIO(b"not an envelope at all"))

    sealed, _ = seal(b"x")
    with pytest.raises(FileDecryptionError, match="truncated"):
        EnvelopeHeader.parse(io.BytesIO(sealed[:20]))

    unsupported = sealed.replace(b'"v":1', b'"v":9')
    with pytest.raises(FileDecryptionError, match="version 9"):
        EnvelopeHeader.parse(io.BytesIO(unsupported))


def test_body_missing_is_refused():
    _, header = seal(b"x")
    with pytest.raises(FileDecryptionError, match="no sealed body"):
        DecryptingReader(io.BytesIO(header.to_bytes()), header, KEY)


def test_short_reads_from_source_are_assembled_into_full_chunks():
    class Trickle(io.RawIOBase):
        def __init__(self, data):
            self.data = io.BytesIO(data)

        def readable(self):
            return True

        def read(self, size=-1):
            return self.data.read(min(3, size) if size and size > 0 else size)

    plaintext = b"0123456789abcdef" * 3 + b"xyz"
    header = EnvelopeHeader.build(16, "test", "k1", b"w")
    sink = io.BytesIO()
    encrypt_stream(Trickle(plaintext), sink, KEY, header)

    with reader(sink.getvalue()) as view:
        assert view.read() == plaintext
