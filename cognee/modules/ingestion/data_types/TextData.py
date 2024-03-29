from typing import BinaryIO
from cognee.infrastructure.data.utils.extract_keywords import extract_keywords
from .IngestionData import IngestionData

def create_text_data(data: str):
    return TextData(data)

class TextData(IngestionData):
    data: str = None

    def __init__(self, data: BinaryIO):
        self.data = data

    def get_identifier(self):
        keywords = extract_keywords(self.data)

        return "text/plain" + "_" + "|".join(keywords)

    def get_data(self):
        return self.data
