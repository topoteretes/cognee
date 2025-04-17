from io import BufferedReader
from typing import Union, BinaryIO, Optional
from .data_types import TextData, BinaryData, S3BinaryData
from tempfile import SpooledTemporaryFile
from s3fs.core import S3File, S3FileSystem
from cognee.modules.ingestion.exceptions import IngestionError


def classify(data: Union[str, BinaryIO], filename: str = None, s3fs: Optional[S3FileSystem] = None):
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        return BinaryData(data, str(data.name).split("/")[-1] if data.name else filename)

    if isinstance(data, S3File):
        derived_filename = str(data.full_name).split("/")[-1] if data.full_name else filename
        return S3BinaryData(s3_path=data.full_name, name=derived_filename, s3=s3fs)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported: {type(data)}"
    )
