from io import BufferedReader
from typing import Union, BinaryIO
from .data_types import TextData, BinaryData
from tempfile import SpooledTemporaryFile

from cognee.modules.ingestion.exceptions import IngestionError


def classify(data: Union[str, BinaryIO], filename: str = None):
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        return BinaryData(
            data, str(data.name).split("/")[-1] if hasattr(data, "name") else filename
        )

    if S3File is not None:
        if isinstance(data, S3File):
            derived_filename = str(data.full_name).split("/")[-1] if data.full_name else filename
            return S3BinaryData(s3_path=data.full_name, name=derived_filename)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported or s3fs is not installed: {type(data)}"
    )
