from io import BytesIO

import pytest

from cognee.infrastructure.files.utils.guess_file_type import guess_file_type


class NamedBytesIO(BytesIO):
    def __init__(self, value: bytes, name: str | None = None, full_name: str | None = None):
        super().__init__(value)
        self._name = name
        self.full_name = full_name

    @property
    def name(self):
        return self._name


@pytest.mark.parametrize(
    ("name", "full_name", "mime_type", "extension"),
    [
        ("s3://bucket/data.json", None, "application/json", "json"),
        (None, "s3://bucket/notes.md", "text/markdown", "md"),
    ],
)
def test_guess_file_type_uses_stream_name_when_name_is_not_passed(
    name, full_name, mime_type, extension
):
    stream = NamedBytesIO(b'{"value": 1}', name=name, full_name=full_name)

    file_type = guess_file_type(stream)

    assert file_type.mime == mime_type
    assert file_type.extension == extension
