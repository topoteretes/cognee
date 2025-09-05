from os import path
from io import BufferedReader
from typing import Union, BinaryIO
from tempfile import SpooledTemporaryFile

from cognee.modules.ingestion.exceptions import IngestionError
from .data_types import TextData, BinaryData, CloudBinaryData


def classify(data: Union[str, BinaryIO], filename: str = None):
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        return BinaryData(data, filename if filename else str(data.name).split("/")[-1])

    try:
        from fsspec.spec import AbstractBufferedFile

        fsspec_file_type = AbstractBufferedFile
    except ImportError:
        fsspec_file_type = None

    # check if data is a fsspec file(s3File, gcsFile, azureFile are subclasses of AbstractBufferedFile)
    if fsspec_file_type is not None:
        if isinstance(data, fsspec_file_type):
            # Other cloud storage don't need to replace "adfs://" with "az://", just for Azure Blob Storage
            full_name = (
                data.full_name.replace("abfs://", "az://")
                if data.full_name.startswith("abfs://")
                else data.full_name
            )

            file_name = path.basename(full_name)

            return CloudBinaryData(cloud_file_path=full_name, name=file_name)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported or s3fs is not installed: {type(data)}"
    )
