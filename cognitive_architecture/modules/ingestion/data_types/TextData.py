from .IngestionData import IngestionData

def create_text_data(data: str):
    return TextData(data)

class TextData(IngestionData):
    data: str = None

    def __init__(self, data: str):
        self.data = data

    def get_data(self):
        return self.data

    def get_chunks(self):
        pass
