from io import BufferedReader
from typing import Union, BinaryIO, Optional, Any
from .data_types import TextData, BinaryData
from tempfile import SpooledTemporaryFile

from cognee.modules.ingestion.exceptions import IngestionError


def classify(data: Union[str, BinaryIO], filename: str = None, s3fs: Optional[Any] = None):
    try:
        from importlib import import_module

        s3core = import_module("s3fs.core")
        S3File = s3core.S3File
    except ImportError:
        S3File = None

    if S3File is not None:
        from .data_types import S3BinaryData

        if isinstance(data, S3File):
            derived_filename = str(data.full_name).split("/")[-1] if data.full_name else filename
            return S3BinaryData(s3_path=data.full_name, name=derived_filename, s3=s3fs)

    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        return BinaryData(data, str(data.name).split("/")[-1] if data.name else filename)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported or s3fs is not installed: {type(data)}"
    )
