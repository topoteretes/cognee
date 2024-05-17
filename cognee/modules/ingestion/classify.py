from io import BufferedReader
from typing import Union, BinaryIO
from .exceptions import IngestionException
from .data_types import TextData, BinaryData

def classify(data: Union[str, BinaryIO], filename: str = None):
    if isinstance(data, str):
        return TextData(data)

    if isinstance(data, BufferedReader):
        return BinaryData(data)

    if hasattr(data, "file"):
        return BinaryData(data.file, filename)

    raise IngestionException(f"Type of data sent to classify(data: Union[str, BinaryIO) not supported: {type(data)}")
