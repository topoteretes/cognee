from io import BufferedReader
from typing import Union, BinaryIO
from .exceptions import IngestionException
from .data_types import create_text_data, create_binary_data

def classify(data: Union[str, BinaryIO]):
    if isinstance(data, str):
        return create_text_data(data)

    if isinstance(data, BufferedReader):
        return create_binary_data(data)

    raise IngestionException(f"Type of data sent to cognee.add(data_path: string | List[string]) not supported: {type(data)}")
