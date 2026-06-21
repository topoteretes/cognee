from os import PathLike, path
from io import BufferedReader
from typing import Union, BinaryIO
from tempfile import SpooledTemporaryFile
from pathlib import Path

from cognee.modules.ingestion.exceptions import IngestionError
from .data_types import TextData, BinaryData, S3BinaryData


def _binary_name(data: BinaryIO, filename: str = None) -> str | None:
    if filename:
        return Path(filename).name

    name = getattr(data, "name", None)
    if isinstance(name, (str, PathLike)):
        return Path(name).name

    return None


def _is_seekable_file_like(data) -> bool:
    return callable(getattr(data, "read", None)) and callable(getattr(data, "seek", None))


def classify(
    data: Union[str, BinaryIO], filename: str = None
) -> Union[TextData, BinaryData, S3BinaryData]:
    if isinstance(data, str):
        return TextData(data)

    try:
        from s3fs import S3File
    except ImportError:
        S3File = None

    if S3File is not None:
        if isinstance(data, S3File):
            return S3BinaryData(s3_path=path.join("s3://", data.bucket, data.key), name=data.key)

    if isinstance(data, (BufferedReader, SpooledTemporaryFile)) or _is_seekable_file_like(data):
        return BinaryData(data, _binary_name(data, filename))

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO]) not supported or s3fs is not installed: {type(data)}"
    )
