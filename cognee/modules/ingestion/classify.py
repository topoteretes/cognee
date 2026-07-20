from os import path
from io import BufferedReader
from typing import Union, BinaryIO
from tempfile import SpooledTemporaryFile

from cognee.modules.ingestion.exceptions import IngestionError
from .data_types import TextData, BinaryData, S3BinaryData


def classify(
    data: Union[str, BinaryIO], filename: str = None
) -> Union[TextData, BinaryData, S3BinaryData]:
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        # Normalize Windows ("\") and POSIX ("/") separators before taking the
        # basename; on Windows `data.name` is a backslash path, so splitting on
        # "/" alone would keep the full path as the file name.
        derived_name = str(data.name).replace("\\", "/").split("/")[-1]
        return BinaryData(data, filename if filename else derived_name)

    try:
        from s3fs import S3File
    except ImportError:
        S3File = None

    if S3File is not None:
        if isinstance(data, S3File):
            return S3BinaryData(s3_path=path.join("s3://", data.bucket, data.key), name=data.key)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported or s3fs is not installed: {type(data)}"
    )
