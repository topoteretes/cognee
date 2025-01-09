from io import BufferedReader
from typing import Union, BinaryIO
from .data_types import TextData, BinaryData
from tempfile import SpooledTemporaryFile

from cognee.modules.ingestion.exceptions import IngestionError


def classify(data: Union[str, BinaryIO], filename: str = None):
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        return BinaryData(data, data.name.split("/")[-1] if data.name else filename)

    raise IngestionError(
        message=f"Type of data sent to classify(data: Union[str, BinaryIO) not supported: {type(data)}"
    )
