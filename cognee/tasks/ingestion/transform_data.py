from llama_index.core import Document
from llama_index.core.schema import ImageDocument
from typing import Union


def from_llama_index_format(data_point: Union[Document, ImageDocument]):
    if type(data_point) == Document:
        return data_point.text
    elif type(data_point) == ImageDocument:
        return data_point.image_path