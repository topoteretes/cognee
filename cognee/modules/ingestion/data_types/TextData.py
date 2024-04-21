from typing import BinaryIO
from cognee.infrastructure.data.utils.extract_keywords import extract_keywords
from .IngestionData import IngestionData

def create_text_data(data: str):
    return TextData(data)

class TextData(IngestionData):
    data: str = None
    metadata = None

    def __init__(self, data: BinaryIO):
        self.data = data

    def get_identifier(self):
        keywords = self.get_metadata()["keywords"]

        return "text/plain" + "_" + "|".join(keywords)

    def get_metadata(self):
        self.ensure_metadata()

        return self.metadata

    def ensure_metadata(self):
        if self.metadata is None:
            self.metadata = dict(keywords = extract_keywords(self.data))

    def get_data(self):
        return self.data
