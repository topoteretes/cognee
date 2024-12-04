from io import BufferedReader
from typing import Union, BinaryIO
from .exceptions import IngestionException
from .data_types import TextData, BinaryData
from tempfile import SpooledTemporaryFile

def classify(data: Union[str, BinaryIO], filename: str = None):
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader) or isinstance(data, SpooledTemporaryFile):
        return BinaryData(data, data.name.split("/")[-1] if data.name else filename)

    raise IngestionException(f"Type of data sent to classify(data: Union[str, BinaryIO) not supported: {type(data)}")
