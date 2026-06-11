"""Background remember must not read request-scoped uploads after teardown.

When ``run_in_background=True`` the HTTP response is sent before ingestion
runs, and Starlette closes each UploadFile's temp file on request teardown.
``_detach_request_files`` copies uploads into pipeline-owned buffers so the
background task survives that.
"""

from tempfile import SpooledTemporaryFile

from cognee.api.v1.remember.remember import _detach_request_files


class UploadFileStub:
    def __init__(self, content: bytes, filename: str):
        self.file = SpooledTemporaryFile()
        self.file.write(content)
        self.file.seek(0)
        self.filename = filename


def test_detached_upload_survives_original_close():
    upload = UploadFileStub(b"hello cognee", "notes.txt")

    detached = _detach_request_files([upload])
    upload.file.close()  # simulates Starlette request teardown

    assert len(detached) == 1
    assert detached[0].filename == "notes.txt"
    assert detached[0].file.read() == b"hello cognee"


def test_detach_preserves_read_position_independence():
    upload = UploadFileStub(b"0123456789", "data.bin")
    upload.file.read(4)  # consumed by e.g. size estimation

    detached = _detach_request_files([upload])

    assert detached[0].file.read() == b"0123456789"


def test_non_upload_items_pass_through_unchanged():
    items = ["plain text", b"bytes too"]

    assert _detach_request_files(items) == items
    assert _detach_request_files("single") == "single"
