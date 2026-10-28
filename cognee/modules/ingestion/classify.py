from io import BufferedReader, BytesIO, IOBase
from os import path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO

from cognee.modules.ingestion.exceptions import IngestionError

from .data_types import BinaryData, S3BinaryData, TextData


def classify(
    data: str | BinaryIO, filename: str | None = None
) -> TextData | BinaryData | S3BinaryData:
    if isinstance(data, str):
        return TextData(data)

    try:
        from s3fs import S3File
    except ImportError:
        S3File = None

    if S3File is not None and isinstance(data, S3File):
        return S3BinaryData(s3_path=path.join("s3://", data.bucket, data.key), name=data.key)

    if isinstance(data, (BufferedReader, SpooledTemporaryFile, BytesIO, IOBase)) or hasattr(
        data, "read"
    ):
        source_name = filename if filename else getattr(data, "name", None)
        # Normalize both POSIX ("/") and Windows ("\") separators before taking the
        # basename. On Windows, data.name is a backslash path (e.g. C:\dir\file.pdf),
        # so splitting on "/" alone would keep the whole path as the file's name.
        derived_name = (
            str(source_name).replace("\\", "/").split("/")[-1] if source_name is not None else None
        )
        return BinaryData(data, derived_name)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported or s3fs is not installed: {type(data)}"
    )
