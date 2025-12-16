from pydantic import BaseModel


class TranscriptionReturnType:
    text: str
    payload: BaseModel

    def __init__(self, text: str, payload: BaseModel):
        self.text = text
        self.payload = payload
